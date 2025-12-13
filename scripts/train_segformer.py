import argparse
import time

import numpy as np

import os
import shutil
import datetime

from tf_keras.src.losses import binary_crossentropy

import data_utils

from railway.shared import data_utils as sh_data_utils, log_utils
from railway.shared import utils as sh_utils

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

import tf_keras

from transformers import TFSegformerForSemanticSegmentation

from tf_keras import backend as K

from pathlib import Path

from transformers.keras_callbacks import KerasMetricCallback
from transformers import AutoImageProcessor

import evaluate
metric = evaluate.load("mean_iou")


class CFG:
    def __init__(self,
                 img_sz, epochs, batch_size, dataset_types):
        self.seed = 42
        self.host_type = sh_utils.detect_host_type()
        self.arch_name = 'segformer'

        self.model_checkpoint = "nvidia/mit-b0"
        self.model_ckpt_name = self.model_checkpoint.replace('/', '_')

        self.mode_direction = 'min'
        self.monitor_metric = 'val_loss'

        self.early_stop = True
        self.stop_patience = 10

        self.lr = 0.00006
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_data = True
        self.tf_cache_dataset = False

        self.dataset_types = dataset_types
        self.data_root = sh_data_utils.resolve_data_root_gen(self.host_type)
        self.dataset_type_paths = sh_data_utils.get_dataset_paths(self.data_root, dataset_types)

        self.image_size = img_sz
        self.mean = tf.constant([0.485, 0.456, 0.406])
        self.std = tf.constant([0.229, 0.224, 0.225])

        self.id2label = {0: "bg", 1: "fg"}
        self.label2id = {label: id for id, label in self.id2label.items()}
        self.num_labels = len(self.id2label)

        self.run_train = True
        self.show_demo_images = False

        self.aug_type = data_utils.AugType.EMPTY
        self.apply_aug = False
        self.use_albumentations = False
        self.aug_horiz_flip = False
        self.aug_vert_flip = False
        self.aug_clahe = False
        self.aug_sat = None #(0.8, 1.2)
        self.aug_cont = None #(0.8, 1.2)
        self.aug_bri = None #0.1

        self.uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
        extra_dir = '' if self.run_train else 'no_train/'

        prefix_dir = 'sh_sd/'
        sh_data_names = [sh_data_utils.get_short_name_by_dataset_type(x).upper() for x in self.dataset_types]
        datatypes_str = '_'.join(sh_data_names)
        weights_dir_name = f'weights_{self.uniq_filename}_{self.model_ckpt_name}_sz{self.image_size}_e{self.epochs}_b{self.batch_size}_{datatypes_str}'
        self.out_dir = f'./tmp/{prefix_dir}{extra_dir}{weights_dir_name}'

        self.split_ratio_train = 0.80
        self.split_ratio_val = 0.10
        self.split_ratio_test = 0.10


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = tf.transpose(logits, perm=[0, 2, 3, 1])
    logits_resized = tf.image.resize(
        logits,
        size=tf.shape(labels)[1:],
        method="bilinear",
    )
    pred_labels = tf.argmax(logits_resized, axis=-1)


    pred_labels = pred_labels.numpy()
    pred_labels = pred_labels.astype(np.int32) # it is int64
    labels = labels.astype(np.int32)

    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=cfg.num_labels,
        ignore_index=-1,
        reduce_labels=image_processor.do_reduce_labels,
    )
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update(
        {f"accuracy_{cfg.id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
    )
    metrics.update({f"iou_{cfg.id2label[i]}": v for i, v in enumerate(per_category_iou)})
    return {"val_" + k: v for k, v in metrics.items()}


def compute_iou_metric(model, image_processor, dataset, num_labels, verbose=False):
    metric = evaluate.load("mean_iou")
    for batch in iter(dataset):
        start_time = time.time()
        predictions = model.predict(batch)
        if verbose:
            log_utils.info('Single exec took {:.4f} secs'.format(time.time() - start_time))

        logits = tf.transpose(predictions.logits, perm=[0, 2, 3, 1])
        logits_resized = tf.image.resize(
            logits,
            size=tf.shape(batch["labels"])[1:],
            method="bilinear",
        )
        pred_labels = tf.argmax(logits_resized, axis=-1)
        metric.add_batch(predictions=pred_labels, references=batch["labels"])

    return metric.compute(
        num_labels=num_labels,
        ignore_index=-1,
        reduce_labels=image_processor.do_reduce_labels,
    )


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def create_train_ds_w_TF_augs(cfg, train_pairs, shuffle=True, cache=True):
    def generator_train():
        for _, image_path, mask_path in train_pairs:
            datapoint = data_utils.load_image_file(image_path, mask_path)
            new_data_point = data_utils.load_resized_normalized_image(datapoint, cfg)
            yield new_data_point

    return data_utils.build_tf_dataset(
        cfg,
        generator_train,
        batch_size=cfg.batch_size,
        apply_augment=cfg.apply_aug,
        shuffle=shuffle,
        cache=cache,
    )


def create_compiled_model(cfg):
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        cfg.model_checkpoint,
        num_labels=cfg.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
        ignore_mismatched_sizes=True,
    )
    optimizer = tf_keras.optimizers.Adam(learning_rate=cfg.lr)
    model.compile(optimizer=optimizer)
    log_utils.info(model.summary())
    return model


def parse_prod_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-sz", "--image_size", type=int, default=None, help="image size")
    ap.add_argument("-e", "--epochs", type=int, default=None, help="epochs")
    ap.add_argument("-b", "--batch_size", type=int, default=None, help="batch size")
    ap.add_argument("-data", '--dataset_types', type=str, default=None, help='Dataset types')
    ap.add_argument('--show_plots', default=True, action=argparse.BooleanOptionalAction)
    return vars(ap.parse_args())


if __name__ == '__main__':
    host_type = sh_utils.detect_host_type()
    log_utils.info(f'{host_type=}')

    args = parse_prod_args()
    log_utils.info(f'{args=}')

    show_plots: bool = args["show_plots"]
    image_size: int = sh_utils.pick_first(args["image_size"], 512)
    batch_size: int = sh_utils.pick_first(args["batch_size"], 8)
    epochs: int = sh_utils.pick_first(args["epochs"], 100)

    if args["dataset_types"] is not None:
        data_types_list = args["dataset_types"].split(',')
        dataset_types = [sh_data_utils.DatasetType(x) for x in data_types_list]
        assert len(dataset_types) > 0, f'Provided dataset types empty {args["dataset_types"]}'
    else:
        dataset_types = [
            sh_data_utils.DatasetType.RSDD_TYPE1_TYPE2_CROP_AUG_BOOST,
            sh_data_utils.DatasetType.SHANGLIAN_DISCR_MCMC_20251205_VAR_4
        ]

    cfg = CFG(
        img_sz=image_size,
        epochs=epochs,
        batch_size=batch_size,
        dataset_types=dataset_types,
    )

    shutil.rmtree(cfg.out_dir, ignore_errors=True)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    log_utils.info('Initialize file saving')
    log_utils.init(cfg.out_dir, filename='logs.log', level="INFO", overwrite=True)
    log_utils.info({x: dict(cfg.__dict__)[x] for x in dict(cfg.__dict__) if
                    not x.startswith('_') and x != 'device' and x != 'mean' and x != 'std'})

    image_processor = AutoImageProcessor.from_pretrained(cfg.model_checkpoint)

    train_pairs, val_pairs, test_pairs = sh_data_utils.resolve_multiple_dataset(cfg.dataset_types, cfg)
    log_utils.info(f'{cfg.dataset_types=}')
    log_utils.info(f'{len(train_pairs)=} {len(val_pairs)=} {len(test_pairs)=}')

    cfg.train_len = len(train_pairs)
    cfg.valid_len = len(val_pairs)
    cfg.test_len = len(test_pairs)
    log_utils.info(f'{len(val_pairs)=} {len(test_pairs)=}')

    model = create_compiled_model(cfg)

    if cfg.apply_aug and cfg.use_albumentations:
        train_ds = data_utils.create_dataset_w_Alb_augs(
            cfg, train_pairs, shuffle=cfg.shuffle_data, cache=cfg.tf_cache_dataset)
    else:
        train_ds = create_train_ds_w_TF_augs(
            cfg, train_pairs, shuffle=cfg.shuffle_data, cache=cfg.tf_cache_dataset)

    valid_ds = data_utils.get_val_dataset(cfg, val_pairs)
    test_ds = data_utils.get_test_dataset(cfg, test_pairs)

    class CustomLogCallback(tf_keras.callbacks.Callback):
        start_time = None
        def on_epoch_begin(self, epoch, logs=None):
            # set time
            self.start_time = time.time()
            return super().on_epoch_begin(epoch, logs)

        def on_epoch_end(self, epoch, logs=None):
            duration = time.time() - self.start_time
            general_info = f"Epoch {epoch + 1}/{self.params['epochs']} - {duration:.3f}s"  # epoch number and time

            loss = ""  # string to hold the losses
            for k, v in logs.items():
                k = k.replace("output_", "")
                if k != "val_loss":
                    k = k.replace("_loss", "")
                # add to string
                loss += f"{k} {v:.4f} - "
            log_utils.info(general_info + ": " + loss)
            return super().on_epoch_end(epoch, logs)


    metric_callback = KerasMetricCallback(
        metric_fn=compute_metrics,
        eval_dataset=valid_ds,
        batch_size=cfg.batch_size,
        label_cols=["labels"],
    )

    checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
        filepath=f"{cfg.out_dir}/best_weights.h5",
        monitor=cfg.monitor_metric,  # Monitor validation loss
        save_best_only=True,  # Save only the best weights
        save_weights_only=True,  # Only save weights, not the entire model
        mode=cfg.mode_direction  # Minimize validation loss
    )

    early_stopping = tf_keras.callbacks.EarlyStopping(
        monitor=cfg.monitor_metric,  # Metric to monitor for improvement
        patience=cfg.stop_patience,          # Number of epochs with no improvement before stopping
        verbose=1,  # Prints messages, including when it restores best weights/stops
        restore_best_weights=True,  # Restore model weights to the epoch with the best value
        mode=cfg.mode_direction,
    )

    sh_utils.save_config(cfg.out_dir, cfg)

    csv_callback = tf_keras.callbacks.CSVLogger(f'{cfg.out_dir}/train_history.csv')
    callback_list = [
        metric_callback,
        checkpoint_callback,
        csv_callback,
        CustomLogCallback()
    ]

    if cfg.early_stop:
        callback_list.append(early_stopping)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        callbacks=callback_list,
        epochs=cfg.epochs,
    )

    eval_loss = model.evaluate(test_ds)
    log_utils.info(f'Test dataset {eval_loss=}')

    weights_dst_file = f"{cfg.out_dir}/segformer_{cfg.model_ckpt_name}.h5"
    log_utils.info('Save weights', weights_dst_file)
    model.save_weights(weights_dst_file)

    weights_dir_out = f"{cfg.out_dir}/out"
    log_utils.info('Save pretrained', weights_dir_out)
    model.save_pretrained(weights_dir_out)
