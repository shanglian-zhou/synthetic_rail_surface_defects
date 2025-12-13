import os
import data_utils

import argparse
from enum import Enum

import cv2
from torchmetrics.classification import MulticlassF1Score

import augs

from railway.shared import data_utils as sh_data_utils, log_utils
from railway.shared import utils as sh_utils


import datetime
import shutil
from pathlib import Path

import numpy as np

from torch.utils.data import Dataset

import albumentations as A
import datasets

from transformers import (
    MaskFormerForInstanceSegmentation, Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
)
import evaluate

from transformers import MaskFormerImageProcessor
from torch.utils.data import DataLoader


import torch
from tqdm.auto import tqdm


class MaskFormerType(str, Enum):
    MASKFORMER = 'maskformer'
    MASK2FORMER = 'mask2former'


class CFG:
    def __init__(self, img_sz: int, batch_size: int, model_type: MaskFormerType, dataset_types: list):
        self.seed = 42
        self.debug = False
        self.host_type = sh_utils.detect_host_type()

        self.model_type = model_type
        self.arch_name = str(self.model_type.value)

        if self.model_type is MaskFormerType.MASKFORMER:
            self.model_name = 'facebook/maskformer-swin-tiny-ade'
        elif self.model_type is MaskFormerType.MASK2FORMER:
            self.model_name = 'facebook/mask2former-swin-tiny-ade-semantic'
        else:
            raise Exception('Invalid model type', self.model_type)

        self.model_ckpt_name = self.model_name.replace('/', '_')

        self.model_cfg_params = {}
        self.model_checkpoint = None

        self.image_size = img_sz

        self.early_stop = True
        self.stop_patience = 5 if self.image_size < 512 else 10

        self.lr = 5e-5
        self.epochs = 100

        self.batch_size = batch_size

        self.shuffle_data = True
        self.is_demo_dataset = False

        self.dataset_types = dataset_types
        self.data_root = sh_data_utils.resolve_data_root_gen(self.host_type)
        assert os.path.exists(self.data_root), f'Not exists {self.data_root}'
        self.dataset_type_paths = sh_data_utils.get_dataset_paths(self.data_root, dataset_types)

        self.mean = np.array([123.675, 116.280, 103.530]) / 255
        self.std = np.array([58.395, 57.120, 57.375]) / 255
        self.mean = self.mean.tolist()
        self.std = self.std.tolist()

        self.ignore_bg_cls = False
        self.bg_cls_id = 1  # 255
        self.id2label = {0: "fg", self.bg_cls_id: "bg"}

        self.label2id = {label: id for id, label in self.id2label.items()}
        self.num_labels = len(self.id2label)

        self.run_train = not self.debug
        self. show_demo_images = False

        self.train_aug_type = data_utils.AugType.EMPTY
        self.aug_da_type = None
        self.apply_aug = False

        self.uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
        self.extra_dir = '' if self.run_train else 'no_train/'

        prefix_dir = 'sh_sd/'
        sh_data_names = [sh_data_utils.get_short_name_by_dataset_type(x).upper() for x in self.dataset_types]
        datatypes_str = '_'.join(sh_data_names)
        weights_dir_name = f'weights_{self.uniq_filename}_{self.model_ckpt_name}_sz{self.image_size}_e{self.epochs}_b{self.batch_size}_{datatypes_str}'
        self.out_dir = f'./tmp/{prefix_dir}{self.extra_dir}{weights_dir_name}'

        self.split_ratio_train = 0.80
        self.split_ratio_val = 0.10
        self.split_ratio_test = 0.10


class ImageSegmentationDemoDataset(Dataset):

    def __init__(self, dataset, transform):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image = np.array(self.dataset[idx]['pixel_values'])
        original_segmentation_map = np.array(self.dataset[idx]['label'])

        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']

        image = image.transpose(2, 0, 1)
        return image, segmentation_map, original_image, original_segmentation_map


class ImageSegmentationRailwayDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, bg_cls_id, item_data_pairs, transform, transform_da=None):
        self.bg_cls_id = bg_cls_id
        self.item_data_pairs = item_data_pairs
        self.transform = transform

        self.transform_da = transform_da

    def __len__(self):
        return len(self.item_data_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.item_data_pairs[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 3 channel
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 1 channel
        mask = np.where(mask == 255, 0, self.bg_cls_id) # FG (class 0), BG (class 1)
        mask = mask.astype(np.uint8)

        original_image = np.array(img)
        original_segmentation_map = np.array(mask)

        applied_da = False
        if self.transform_da is not None:
            is_train_imgs_ref = 'rsdds_type1' in image_path or 'rsdds_type2' in image_path
            if not is_train_imgs_ref: # synthetic image?
                transformed_image = self.transform_da(image=original_image)["image"]
                transformed = self.transform(image=transformed_image, mask=original_segmentation_map)
                applied_da = True

        if not applied_da:
            transformed = self.transform(image=original_image, mask=original_segmentation_map)

        image, segmentation_map = transformed['image'], transformed['mask']
        image = image.transpose(2, 0, 1)
        return image, segmentation_map, original_image, original_segmentation_map


def validation_step(cfg, model, preprocessor, data_dataloader, id2label, device):
    metric = evaluate.load("mean_iou")
    val_loss = []
    running_dice = 0.0
    model.eval()

    for idx, batch in enumerate(tqdm(data_dataloader)):
        if cfg.is_demo_dataset and idx > 5:
            break

        pixel_values = batch["pixel_values"]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values.to(device))

        original_images = batch["original_images"]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
        # predict segmentation maps
        predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                      target_sizes=target_sizes)
        # get ground truth segmentation maps
        ground_truth_segmentation_maps = batch["original_segmentation_maps"]

        metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)

        dice = MulticlassF1Score(num_classes=cfg.num_labels, average=None)
        for pred_mask, gt_mask in zip(predicted_segmentation_maps, ground_truth_segmentation_maps):
            dice_score_per_class = dice(pred_mask.detach().cpu().clone(), torch.from_numpy(gt_mask))
            dice_score_fg = dice_score_per_class[0]
            running_dice += dice_score_fg.item()

    mean_dice = running_dice / len(data_dataloader.dataset)

    ignore_index = cfg.bg_cls_id if cfg.ignore_bg_cls else -1
    metrics = metric.compute(
        num_labels=len(id2label), # remove background class
        ignore_index=ignore_index,
        reduce_labels=False,
    )

    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()
    metrics.update(
        {f"accuracy_{cfg.id2label[i]}": v for i, v in enumerate(per_category_accuracy) if i in cfg.id2label}
    )
    metrics.update({f"iou_{cfg.id2label[i]}": v for i, v in enumerate(per_category_iou) if i in cfg.id2label})
    metrics.update({'dice': mean_dice})
    return val_loss, metrics


def train_step(model, optimizer, train_dataloader, device=torch.device('cuda')):
    step_loss = []
    model.train()

    running_loss = 0.0
    num_samples = 0

    for idx, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        if idx % 100 == 0:
            log_utils.info("Training loss:", running_loss / num_samples)
        step_loss.append(loss.item())
        optimizer.step()

    return step_loss, running_loss, num_samples


def run_train(cfg, model, preprocessor, train_dataloader, test_dataloader, id2label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.early_stop:
        early_stopper = EarlyStopperHigherBetter(patience=cfg.stop_patience, min_delta=0.00001)
    else:
        early_stopper = None

    best_loss = float('-inf')
    running_loss = 0.0
    num_samples = 0

    trainingEpoch_loss = []
    validationEpoch_loss = []
    val_iou_metrics_history = []

    for epoch in range(cfg.epochs):
        log_utils.info("Epoch:", epoch, "| Training")

        # TRAIN
        step_loss, running_loss_, num_samples_ = train_step(model, optimizer, train_dataloader, device=device)
        running_loss += running_loss_
        num_samples += num_samples_
        trainingEpoch_loss.append(np.array(step_loss).mean())

        # VALIDATION
        validation_loss, iou_metrics_dict = validation_step(
            cfg, model, preprocessor, test_dataloader, id2label, device=device)
        mean_iou = iou_metrics_dict['mean_iou']
        log_utils.info("VAL loss: ", validation_loss, iou_metrics_dict)

        val_iou_metrics_history.append(iou_metrics_dict)
        validationEpoch_loss.append(mean_iou)
        watch_metric = mean_iou

        if watch_metric > best_loss:
            log_utils.info(f'mean IoU increased {best_loss} -> {watch_metric}')
            best_loss = watch_metric
            weights_dir_out = f"{cfg.out_dir}/out_best_loss"
            save_weights_to_dir(model, preprocessor, weights_dir_out)

        if early_stopper is not None and early_stopper.early_stop(mean_iou):
            log_utils.info(f'Early stopping triggered at epoch {epoch}. Loss not decreasing')
            break

    return model, trainingEpoch_loss, val_iou_metrics_history


def create_dataset(image_paths, label_paths):
    dataset = datasets.Dataset.from_dict({"pixel_values": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("pixel_values", datasets.Image())
    dataset = dataset.cast_column("label", datasets.Image())

    def transforms(examples):
        examples["label"] = [mask.convert("LA") for mask in examples["label"]]
        return examples
    dataset = dataset.map(transforms, batched=True)
    return dataset


def load_railway_dataset(cfg):
    log_utils.info('Loading railway dataset')
    train_pairs, val_pairs, test_pairs = sh_data_utils.resolve_multiple_dataset(cfg.dataset_types, cfg)
    log_utils.info(f'{cfg.dataset_types=}')
    log_utils.info(f'{len(train_pairs)=} {len(val_pairs)=} {len(test_pairs)=}')

    train_pairs_matched = [(image_path, mask_path) for _, image_path, mask_path in train_pairs]
    val_pairs_matched = [(image_path, mask_path) for _, image_path, mask_path in val_pairs]
    test_pairs_matched = [(image_path, mask_path) for _, image_path, mask_path in test_pairs]

    train_transform, extra_args = augs.get_train_augs_general(cfg, train_pairs)
    log_utils.info(f'{train_transform=}')

    val_transform = A.Compose([
        A.Resize(width=cfg.image_size, height=cfg.image_size),
        A.Normalize(mean=cfg.mean, std=cfg.std),
    ])
    log_utils.info(f'{val_transform=}')

    test_transform = A.Compose([
        A.Resize(width=cfg.image_size, height=cfg.image_size),
        A.Normalize(mean=cfg.mean, std=cfg.std),
    ])
    log_utils.info(f'{test_transform=}')

    train_dataset = ImageSegmentationRailwayDataset(
        cfg.bg_cls_id,
        train_pairs_matched,
        transform=train_transform,
        **extra_args
    )
    val_dataset = ImageSegmentationRailwayDataset(cfg.bg_cls_id, val_pairs_matched, transform=val_transform)
    test_dataset = ImageSegmentationRailwayDataset(cfg.bg_cls_id, test_pairs_matched, transform=test_transform)
    return train_dataset, val_dataset, test_dataset


class EarlyStopperHigherBetter:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_loss = float('-inf')

    def early_stop(self, validation_loss):
        if validation_loss > self.max_validation_loss:
            self.max_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def save_weights_to_dir(model, preprocessor, write_dir):
    shutil.rmtree(write_dir, ignore_errors=True)
    Path(write_dir).mkdir(parents=True, exist_ok=True)

    log_utils.info('Save weights', write_dir)
    model.save_pretrained(os.path.join(write_dir, 'weights'))
    preprocessor.save_pretrained(os.path.join(write_dir, 'preprocessor'))


def parse_prod_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-sz", "--image_size", type=int, default=None,
                    help="image size")
    ap.add_argument("-e", "--epochs", type=int, default=None,
                    help="epochs")
    ap.add_argument("-b", "--batch_size", type=int, default=None,
                    help="batch size")
    ap.add_argument("-data", '--dataset_types', type=str, default=None, help='Dataset types')
    ap.add_argument("-model", '--model_type', type=str, default=None, help='Model type')
    return vars(ap.parse_args())


def run_main():
    host_type = sh_utils.detect_host_type()
    log_utils.info(f'{host_type=}')

    args = parse_prod_args()
    log_utils.info(f'{args=}')

    image_size = sh_utils.pick_first(args["image_size"], 512)
    batch_size = sh_utils.pick_first(args["batch_size"], None)

    if args["dataset_types"] is not None:
        data_types_list = args["dataset_types"].split(',')
        dataset_types = [sh_data_utils.DatasetType(x) for x in data_types_list]
        assert len(dataset_types) > 0, f'Provided dataset types empty {args["dataset_types"]}'
    else:
        dataset_types = [
            sh_data_utils.DatasetType.RSDD_TYPE1_TYPE2_CROP_AUG_BOOST,
            sh_data_utils.DatasetType.SHANGLIAN_DISCR_MCMC_20251205_VAR_4
        ]

    model_type = sh_utils.pick_first(args["model_type"], MaskFormerType.MASKFORMER)
    model_type = MaskFormerType(model_type)

    if batch_size is None:
        batch_size = 8

    cfg = CFG(
        img_sz=image_size,
        batch_size=batch_size,
        dataset_types=dataset_types,
        model_type=model_type,
    )

    shutil.rmtree(cfg.out_dir, ignore_errors=True)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    log_utils.init(cfg.out_dir, filename='logs.log', level="INFO", overwrite=True)
    log_utils.info({x: dict(cfg.__dict__)[x] for x in dict(cfg.__dict__) if
                    not x.startswith('_') and x != 'device' and x != 'mean' and x != 'std'})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = load_railway_dataset(cfg)

    image, segmentation_map, _, _ = train_dataset[0]
    log_utils.info(image.shape)
    log_utils.info(segmentation_map.shape)

    labels = [cfg.id2label[label] for label in np.unique(segmentation_map) if label in cfg.id2label]
    log_utils.info(f'{labels=}')

    ignore_idx = cfg.bg_cls_id if cfg.ignore_bg_cls else -1
    log_utils.info(f'{ignore_idx=}')

    if cfg.model_type is MaskFormerType.MASKFORMER:
        preprocessor = MaskFormerImageProcessor(
            ignore_index=ignore_idx, # ignore background
            reduce_labels=False,
            do_resize=False,
            do_rescale=False,
            do_normalize=False
        )
    elif cfg.model_type is MaskFormerType.MASK2FORMER:
        preprocessor = Mask2FormerImageProcessor(
            ignore_index=ignore_idx, # ignore background
            reduce_labels=False,
            do_resize=False,
            do_rescale=False,
            do_normalize=False
        )
    else:
        raise Exception('Invalid model type', cfg.model_type)

    def collate_fn(batch):
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]
        batch = preprocessor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )
        batch["original_images"] = inputs[2]
        batch["original_segmentation_maps"] = inputs[3]
        return batch


    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    if cfg.is_demo_dataset:
        test_dataloader = None
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(train_dataloader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            log_utils.info(k, v.shape)
        else:
            log_utils.info(k, v[0].shape)

    if cfg.model_type is MaskFormerType.MASKFORMER:
        if cfg.model_checkpoint is None:
            log_utils.info('Loading pretrained model')
            model = MaskFormerForInstanceSegmentation.from_pretrained(
                cfg.model_name,
                num_labels=cfg.num_labels,
                id2label=cfg.id2label,
                label2id=cfg.label2id,
                ignore_mismatched_sizes=True,
                **cfg.model_cfg_params
            )
        else:
            log_utils.info('Loading fine-tuned model')
            model = MaskFormerForInstanceSegmentation.from_pretrained(os.path.join(cfg.model_checkpoint, 'weights'))
    elif cfg.model_type is MaskFormerType.MASK2FORMER:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_labels,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
            ignore_mismatched_sizes=True
        )
    else:
        raise Exception('Invalid model type', cfg.model_type)

    outputs = model(batch["pixel_values"],
                    class_labels=batch["class_labels"],
                    mask_labels=batch["mask_labels"])
    log_utils.info('Experimental loss', outputs.loss)

    model, _, _ =\
        run_train(cfg, model, preprocessor, train_dataloader, val_dataloader, cfg.id2label)

    weights_dir_out = f"{cfg.out_dir}/out"
    save_weights_to_dir(model, preprocessor, weights_dir_out)

    test_loss, test_iou_metrics = validation_step(cfg, model, preprocessor, test_dataloader, cfg.id2label, device=device)
    log_utils.info("Test loss: ", test_loss, test_iou_metrics)


if __name__ == '__main__':
    run_main()
