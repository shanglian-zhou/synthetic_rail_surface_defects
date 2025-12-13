import argparse
import datetime
import shutil
import time
from pathlib import Path
import tf_keras
import augs
import data_utils
from railway.shared import data_utils as sh_data_utils, log_utils
from railway.shared import utils as sh_utils
from railway.unet import unet_utils

import os
import cv2

import tensorflow as tf
import numpy as np

from tf_keras.callbacks import ModelCheckpoint

from tf_keras import backend, layers, models, utils
import efficientnet as ef
ef._KERAS_BACKEND = backend
ef._KERAS_LAYERS = layers
ef._KERAS_MODELS = models
ef._KERAS_UTILS = utils

import efficientnet.keras as efn

_ = efn.EfficientNetB0(weights='imagenet')

import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()

import albumentations as A


sm_archs = {'unet': sm.Unet, 'fpn': sm.FPN, 'linknet': sm.Linknet}

class CFG:

    def __init__(self,
                 img_sz, epochs, batch_size, dataset_types, arch, backbone, old_scheme=False):
        self.seed = 42
        self.host_type = sh_utils.detect_host_type()
        self.gpu_devices = tf.config.list_physical_devices('GPU')

        self.old_scheme = old_scheme
        self.arch = arch
        if self.arch not in sm_archs:
            raise Exception('Unsupported arch', self.arch)

        self.backbone = backbone
        self.model_ckpt_name = self.backbone

        self.CLASSES = [255]
        self.n_classes = 1 if len(self.CLASSES) == 1 else (len(self.CLASSES) + 1)
        self.activation = 'sigmoid' if self.n_classes == 1 else 'softmax'

        if self.old_scheme:
            self.loss = 'bce_dice_loss'
        else:
            self.loss = 'dice_and_focal_loss'
        self.mode_direction = 'min'
        self.monitor_metric = 'val_loss'

        self.early_stop = True
        self.stop_patience = 10

        self.image_size = img_sz

        self.lr = 0.0001
        self.epochs = epochs
        self.batch_size = batch_size

        self.dataset_types = dataset_types
        self.data_root = sh_data_utils.resolve_data_root_gen(self.host_type)

        assert os.path.exists(self.data_root), f'Not exists {self.data_root}'
        self.dataset_type_paths = sh_data_utils.get_dataset_paths(self.data_root, self.dataset_types)

        self.run_train = True
        self.show_demo_images = False

        self.aug_type = data_utils.AugType.EMPTY
        self.apply_aug = False

        prefix_dir = 'sh_sd/'
        sh_data_names = [sh_data_utils.get_short_name_by_dataset_type(x).upper() for x in self.dataset_types]
        datatypes_str = '_'.join(sh_data_names)
        uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
        weights_dir_name = f'weights_{uniq_filename}_{self.arch}_{self.model_ckpt_name}_sz{self.image_size}_e{self.epochs}_b{self.batch_size}_{datatypes_str}'
        self.out_dir = f'./tmp/{prefix_dir}{weights_dir_name}'

        self.split_ratio_train = 0.8
        self.split_ratio_val = 0.10
        self.split_ratio_test = 0.10


class CustomLogCallback(tf_keras.callbacks.Callback):
    start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        return super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        general_info = f"Epoch {epoch + 1}/{self.params['epochs']} - {duration:.3f}s"  # epoch number and time

        loss = ""
        for k, v in logs.items():
            k = k.replace("output_", "")
            if k != "val_loss":
                k = k.replace("_loss", "")
            # add to string
            loss += f"{k} {v:.4f} - "
        log_utils.info(general_info + ": " + loss)
        return super().on_epoch_end(epoch, logs)


class Dataset:
    def __init__(
            self,
            data_pairs,
            img_size,
            class_idx_arr=None,
            augmentation=None,
            preprocessing=None,
            ret_gt_img_mask_paths=False,
    ):
        self.ids = [x[0] for x in data_pairs] # file stems
        self.images_fps = [x[1] for x in data_pairs] # image list
        self.masks_fps = [x[2] for x in data_pairs] # mask list

        self.img_size = img_size
        self.class_values = class_idx_arr

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.ret_gt_img_mask_paths = ret_gt_img_mask_paths

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        dsize = (self.img_size, self.img_size)
        image = cv2.resize(image, dsize)
        mask = cv2.resize(mask, dsize)
        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.ret_gt_img_mask_paths:
            return image, mask, self.images_fps[i], self.masks_fps[i]
        else:
            return image, mask

    def __len__(self):
        return len(self.ids)


class ImageDatasetLoader(tf_keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def create_model(cfg):
    model = sm_archs[cfg.arch](
        cfg.backbone,
        classes=cfg.n_classes,
        activation=cfg.activation,
        encoder_weights='imagenet',

        backend=backend,
        layers=layers,
        models=models,
        utils=utils,
    )
    if cfg.loss == 'bce_dice_loss':
        use_loss = unet_utils.bce_dice_loss
    elif cfg.loss == 'binary_crossentropy':
        use_loss = 'binary_crossentropy'
    elif cfg.loss == 'dice_and_focal_loss':
        use_loss = unet_utils.get_dice_and_focal_loss()
    else:
        raise Exception(f'Loss {cfg.loss} is not supported')

    model.summary()
    model.compile(
        optimizer='adam',
        loss=use_loss,
        metrics=unet_utils.get_metrics_list()
    )
    return model


def parse_prod_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-sz", "--image_size", type=int, default=None,
                    help="image size")
    ap.add_argument("-e", "--epochs", type=int, default=None,
                    help="epochs")
    ap.add_argument("-b", "--batch_size", type=int, default=None,
                    help="batch size")
    ap.add_argument("-data", '--dataset_types', type=str, default=None, help='Dataset types')
    ap.add_argument('--arch', type=str, default=None, help='arch')
    ap.add_argument('--backbone', type=str, default=None, help='backbone')
    ap.add_argument('--show_plots', default=True, action=argparse.BooleanOptionalAction)
    args = vars(ap.parse_args())
    return args


def run_main():
    host_type = sh_utils.detect_host_type()
    log_utils.info(f'{host_type=}')

    args = parse_prod_args()
    log_utils.info(f'{args=}')

    image_size = sh_utils.pick_first(args["image_size"], 512)
    batch_size = sh_utils.pick_first(args["batch_size"], 16)
    epochs = sh_utils.pick_first(args["epochs"], 100)

    if args["dataset_types"] is not None:
        data_types_list = args["dataset_types"].split(',')
        dataset_types = [sh_data_utils.DatasetType(x) for x in data_types_list]
        assert len(dataset_types) > 0, f'Provided dataset types empty {args["dataset_types"]}'
    else:
        dataset_types = [
            sh_data_utils.DatasetType.RSDD_TYPE1_TYPE2_CROP_AUG_BOOST,
            sh_data_utils.DatasetType.SHANGLIAN_DISCR_MCMC_20251205_VAR_4
        ]

    old_scheme = False
    backbone = 'efficientnetb2'
    backbone = sh_utils.pick_first(args["backbone"], backbone)
    arch = sh_utils.pick_first(args["arch"], 'unet')
    log_utils.info(f'{arch=} {backbone=}')

    cfg = CFG(
        img_sz=image_size,
        epochs=epochs,
        batch_size=batch_size,
        dataset_types=dataset_types,
        arch=arch,
        backbone=backbone,
        old_scheme=old_scheme,
    )

    shutil.rmtree(cfg.out_dir, ignore_errors=True)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    log_utils.info('Initialize file saving')
    log_utils.init(cfg.out_dir, filename='logs.log', level="INFO", overwrite=True)
    log_utils.info({x: dict(cfg.__dict__)[x] for x in dict(cfg.__dict__) if
                    not x.startswith('_') and x != 'device' and x != 'mean' and x != 'std'})

    train_pairs, val_pairs, test_pairs = sh_data_utils.resolve_multiple_dataset(cfg.dataset_types, cfg)
    preprocess_input = sm.get_preprocessing(cfg.backbone)

    model = create_model(cfg)

    checkpoint = ModelCheckpoint(
        f'{cfg.out_dir}/best_model.h5',
        monitor=cfg.monitor_metric,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode=cfg.mode_direction,
    )
    sh_utils.save_config(cfg.out_dir, cfg)

    if cfg.aug_type is data_utils.AugType.EMPTY:
        train_aug_transforms = None
    elif cfg.aug_type is data_utils.AugType.HEAVY:
        train_aug_transforms = augs.get_train_heavy_augmentation(image_size=cfg.image_size)
    elif cfg.aug_type is data_utils.AugType.CLAHE_ONLY:
        train_aug_transforms = augs.get_train_clahe_only_augmentation()
    else:
        raise Exception('Augmentation type is not supported', cfg.aug_type)

    train_dataset = Dataset(
        data_pairs=train_pairs,
        img_size=cfg.image_size,
        class_idx_arr=[255],
        augmentation=train_aug_transforms,
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dataset = Dataset(
        data_pairs=val_pairs,
        img_size=cfg.image_size,
        class_idx_arr=[255],
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = ImageDatasetLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dataloader = ImageDatasetLoader(valid_dataset, batch_size=1, shuffle=False)

    csv_callback = tf_keras.callbacks.CSVLogger(f'{cfg.out_dir}/train_history.csv')
    reduce_LR_callback = tf_keras.callbacks.ReduceLROnPlateau()

    callback_list = [
        checkpoint,
        reduce_LR_callback,
        csv_callback,
        CustomLogCallback()
    ]

    if cfg.early_stop:
        early_stopping = tf_keras.callbacks.EarlyStopping(
            monitor=cfg.monitor_metric,  
            patience=cfg.stop_patience,  
            restore_best_weights=True, 
            mode=cfg.mode_direction,
            verbose=1,
        )
        callback_list.append(early_stopping)

    _ = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
        callbacks=callback_list,
        epochs=cfg.epochs
    )
    weights_dst_file = f"{cfg.out_dir}/{cfg.arch}_{cfg.model_ckpt_name}_entire.h5"
    model.save(weights_dst_file)


if __name__ == '__main__':
    run_main()
