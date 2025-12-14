import datetime
import os
import shutil
from pathlib import Path
from typing import Final

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from railway import config
from railway.shared import utils, log_utils
from railway.var_auto_encoder import vae_utils
from railway.var_auto_encoder.vae_utils import RealDefectDatasetType, TrainImageMaskDataset, LossItem, KlAnnealSchedule, \
    LossType, BetaType


class CFG:
    def __init__(
            self,
            debug,
            host_type,

            defect_data_type: RealDefectDatasetType,
            img_dir: str,
            mask_dir: str,

            latent_dim: int,
            base_ch: int,

            epochs: int,
            batch_size: int,
            learning_rate: float,
            dataloader_workers: int,

            loss_item: LossItem,
            skip_drop_prob: float,

            img_size: int,
            out_write_dir: str
    ):
        self.seed = 42
        self.debug = debug
        self.host_type = host_type

        self.defect_data_type = defect_data_type
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.latent_dim = latent_dim
        self.base_ch = base_ch

        self.is_cuda_enabled = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_enabled else "cpu")
        self.device_count = torch.cuda.device_count()

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataloader_workers = dataloader_workers

        self.loss_item = loss_item
        self.skip_drop_prob = skip_drop_prob

        self.img_size = img_size

        self.uniq_date_filename = str(datetime.datetime.now().date()) + '_' + str(
            datetime.datetime.now().time()).replace(':', '.')
        extra_dir = 'debug_' if self.debug else ''
        data_type = str(self.defect_data_type).lower()
        postfix = f'_{self.uniq_date_filename}_{data_type}_e{self.epochs}_b{self.batch_size}_sz{self.img_size}'
        self.out_dir = os.path.join(out_write_dir, f'{extra_dir}weights{postfix}')


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv_block(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        return y, self.pool(y)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetVAE(nn.Module):
    def __init__(self, in_ch=4, base_ch=32, latent_dim=64, skip_drop_prob=0.0):
        super().__init__()
        self.base_ch = base_ch
        self.latent_dim = latent_dim
        self.skip_drop_prob = skip_drop_prob

        self.down1 = Down(in_ch, base_ch)          # base_ch
        self.down2 = Down(base_ch, base_ch*2)      # 2*base_ch
        self.down3 = Down(base_ch*2, base_ch*4)    # 4*base_ch
        self.bottleneck = conv_block(base_ch*4, base_ch*8)
        self.conv_mu = nn.Conv2d(base_ch*8, latent_dim, 1)
        self.conv_logvar = nn.Conv2d(base_ch*8, latent_dim, 1)

        self.z_to_feat = nn.Conv2d(latent_dim, base_ch*8, 1)

        self.up3 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up1 = Up(base_ch*2, base_ch)
        self.final_conv = nn.Conv2d(base_ch, 3, 1)

    def encode(self, img, mask):
        x = torch.cat([img, mask], dim=1)  # (B,4,H,W)
        s1, p1 = self.down1(x)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        b = self.bottleneck(p3)
        mu = self.conv_mu(b)
        logvar = self.conv_logvar(b)
        return mu, logvar, (s1, s2, s3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _apply_skip_dropout(self, skips: tuple):
        s1, s2, s3 = skips
        if not self.training or self.skip_drop_prob <= 0.0:
            return skips
        new_skips = []
        for s in (s1, s2, s3):
            if torch.rand(1).item() < self.skip_drop_prob:
                new_skips.append(torch.zeros_like(s))
            else:
                new_skips.append(s)
        return tuple(new_skips)

    def decode(self, z, skips):
        skips_for_decode = self._apply_skip_dropout(skips)
        s1, s2, s3 = skips_for_decode
        feat = self.z_to_feat(z)  # (B, base_ch*8, h, w)
        x = feat
        x = self.up3(x, s3)   # -> base_ch*4
        x = self.up2(x, s2)   # -> base_ch*2
        x = self.up1(x, s1)   # -> base_ch
        out = torch.sigmoid(self.final_conv(x))
        return out

    def forward(self, img, mask):
        mu, logvar, skips = self.encode(img, mask)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, skips)
        return out, mu, logvar


def unet_vae_loss_l1(recon, img, mu, logvar, beta=1.0):
    recon_loss = F.l1_loss(recon, img)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


def sample_and_save_unet(
        model: nn.Module, masks: torch.Tensor, device: torch.device, out_path: str, epoch: int, n_per_mask=4):
    model.eval()
    with torch.no_grad():
        for i in range(masks.size(0)):
            mask = masks[i:i+1].to(device)  # real mask
            dummy_img = torch.zeros_like(mask).repeat(1, 3, 1, 1).to(device)
            mu, logvar, skips = model.encode(dummy_img, mask)
            for s in range(n_per_mask):
                z = torch.randn_like(mu)  # sample from standard normal prior
                synth = model.decode(z, skips)
                im_path = os.path.join(out_path, f"unet_sample_e{epoch}_mask{i}_s{s}.png")
                save_image(synth, im_path, normalize=False)


def train_loop(
        model: nn.Module,
        dataloader,
        epochs=10,
        lr=1e-4,
        loss_item: LossItem = LossItem.create_default(),
        out_dir=None,
        device=torch.device('cuda')
):
    recon_save_dir = os.path.join(out_dir, "unet_recon")
    os.makedirs(recon_save_dir, exist_ok=True)
    sample_save_dir = os.path.join(out_dir, "unet_samples")
    os.makedirs(sample_save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc="Train unet cvae", position=0, leave=True)
        total_loss = 0.0
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            recon, mu, logvar = model(imgs, masks)
            loss_type = loss_item.loss_type
            beta_type = loss_item.beta_type

            if beta_type is BetaType.BASELINE:
                beta = loss_item.beta
            elif beta_type is BetaType.KL_ANNEAL_SCHEDULE:
                kl_schedule = loss_item.kl_anneal_schedule
                beta = vae_utils.get_beta(
                    epoch=epoch,
                    kl_anneal_epochs=kl_schedule.kl_anneal_epochs,
                    beta_start=kl_schedule.beta_start,
                    beta_end=kl_schedule.beta_end
                )
            else:
                raise Exception('Invalid arg', beta_type)

            if loss_type is LossType.L1_LOSS:
                loss, rec_loss, kld = unet_vae_loss_l1(recon, imgs, mu, logvar, beta=beta)
            elif loss_type is LossType.SPATIAL_SUM_FREE_BITS:
                kld = vae_utils.compute_kld(mu, logvar, free_bits=loss_item.free_bits, sum_dim=(1, 2, 3))
                rec_loss = F.l1_loss(recon, imgs)
                loss = rec_loss + beta * kld
            else:
                raise Exception('Invalid arg', loss_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch}, loss:{total_loss/(pbar.n+1):.4f} rec:{rec_loss:.4f} kld:{kld:.4f}")

        n_img_eval = 4
        model.eval()
        with torch.no_grad():
            small_imgs = imgs[:n_img_eval]
            small_masks = masks[:n_img_eval]
            recon, _, _ = model(small_imgs, small_masks)
            grid = torch.cat([small_imgs, recon, small_masks.repeat(1, 3, 1, 1)], dim=0)
            out_im_path = os.path.join(recon_save_dir, f"unet_recon_ep{epoch:03d}.png")
            save_image(grid, out_im_path, nrow=n_img_eval, normalize=False)
            sample_and_save_unet(model, small_masks.cpu(), device, sample_save_dir, epoch=epoch, n_per_mask=3)
    return model


def train_unet_cvae(cfg, device=torch.device('cuda')):
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    im_mask_dataset = TrainImageMaskDataset(cfg.img_dir, cfg.mask_dir, img_size=cfg.img_size)
    dataloader = DataLoader(
        im_mask_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.dataloader_workers, pin_memory=True)
    log_utils.info('Total files in dataset', len(im_mask_dataset))

    model = UNetVAE(
        in_ch=4, base_ch=cfg.base_ch, latent_dim=cfg.latent_dim, skip_drop_prob=cfg.skip_drop_prob).to(device)
    model = train_loop(
        model,
        dataloader=dataloader,
        epochs=cfg.epochs,
        lr=cfg.learning_rate,
        loss_item=cfg.loss_item,
        out_dir=out_dir,
        device=device
    )
    ckpt_vae_out_path = os.path.join(out_dir, "unet_cvae_mask2img.pth")
    torch.save(model.state_dict(), ckpt_vae_out_path)
    log_utils.info("Training finished. Models & samples in:", cfg.out_dir)


def run_main():
    host_type = utils.detect_host_type()
    log_utils.info(f'{host_type=}')

    defect_data_type = RealDefectDatasetType.TYPE1_CROP
    img_dir, mask_dir = vae_utils.get_real_mask_image_dirs(host_type, defect_data_type)

    out_write_dir: Final[Path] = Path(config.TMP_FILES_DIR) / 'unet_cvae'
    os.makedirs(out_write_dir, exist_ok=True)

    debug = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 256
    latent_dim = 128
    base_ch = 32
    batch_size = 8
    num_workers = 4
    epochs = 150
    lr = 1e-4
    skip_drop_prob = 0.2

    loss_type = LossType.SPATIAL_SUM_FREE_BITS
    beta_type = BetaType.KL_ANNEAL_SCHEDULE
    beta = 1.0
    free_bits = 0.1

    if beta_type is BetaType.KL_ANNEAL_SCHEDULE:
        beta_start = 0.0  # Initial KL weight
        beta_end = 1.0  # Final KL weight
        kl_anneal_epochs = 50  # Number of epochs to anneal KL
        kl_anneal_schedule = KlAnnealSchedule(
            beta_start=beta_start,
            beta_end=beta_end,
            kl_anneal_epochs=kl_anneal_epochs
        )
    else:
        kl_anneal_schedule = None
    loss_item = LossItem(
        loss_type=loss_type,
        beta_type=beta_type,
        beta=beta,
        free_bits=free_bits,
        kl_anneal_schedule=kl_anneal_schedule
    )

    cfg = CFG(
        debug=debug,
        host_type=host_type,
        defect_data_type=defect_data_type,
        img_dir=img_dir,
        mask_dir=mask_dir,
        latent_dim=latent_dim,
        base_ch=base_ch,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        dataloader_workers=num_workers,
        loss_item=loss_item,
        skip_drop_prob=skip_drop_prob,
        img_size=img_size,
        out_write_dir=str(out_write_dir)
    )
    shutil.rmtree(cfg.out_dir, ignore_errors=True)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    log_level = "DEBUG" if cfg.debug else "INFO"
    log_utils.init(cfg.out_dir, filename='logs.log', level=log_level, overwrite=True)

    train_unet_cvae(cfg, device=device)


if __name__ == "__main__":
    run_main()
