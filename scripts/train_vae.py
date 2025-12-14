import shutil
from pathlib import Path
from typing import Final

import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm

from railway.shared import utils, log_utils
from railway import config
from railway.var_auto_encoder import vae_utils
from railway.var_auto_encoder.vae_utils import TrainImageMaskDataset, RealDefectDatasetType, LossType


class CFG:
    def __init__(
            self,
            debug,
            host_type,

            defect_data_type: RealDefectDatasetType,
            img_dir: str,
            mask_dir: str,

            latent_dim: int,

            epochs: int,
            batch_size: int,
            learning_rate: float,
            dataloader_workers: int,
            img_size: int,
            loss_type: LossType,
            beta: float,
            out_write_dir: str
    ):
        self.seed = 42
        self.debug = debug
        self.host_type = host_type

        self.defect_data_type = defect_data_type
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.latent_dim = latent_dim

        self.is_cuda_enabled = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_enabled else "cpu")
        self.device_count = torch.cuda.device_count()

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataloader_workers = dataloader_workers

        self.img_size = img_size
        self.loss_type = loss_type
        self.beta = beta

        self.uniq_date_filename = str(datetime.datetime.now().date()) + '_' + str(
            datetime.datetime.now().time()).replace(':', '.')
        extra_dir = 'debug_' if self.debug else ''
        data_type = str(self.defect_data_type).lower()
        postfix = f'_{self.uniq_date_filename}_{data_type}_e{self.epochs}_b{self.batch_size}_sz{self.img_size}'
        self.out_dir = os.path.join(out_write_dir, f'{extra_dir}weights{postfix}')


class Encoder(nn.Module):
    def __init__(self, in_ch=4, latent_dim=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.ReLU(),       # H/2
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),   # H/4
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),  # H/8
        )
        self.conv_mu = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(256, latent_dim, kernel_size=1)

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)  # (B,4,H,W)
        h = self.down(x)
        mu = self.conv_mu(h)  # center of the distribution
        logvar = self.conv_logvar(h)  # logarithm of the variance
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, mask_ch=1, latent_dim=128):
        super().__init__()
        # downsample mask to encoder spatial size (H/8)
        self.mask_down = nn.Sequential(
            nn.Conv2d(mask_ch, 64, 4, 2, 1), nn.ReLU(),     # H/2
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),         # H/4
            nn.Conv2d(128, latent_dim, 4, 2, 1), nn.ReLU()  # H/8, -> (latent_dim, H/8, W/8)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 256, 4, 2, 1), nn.ReLU(),   # H/4
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),               # H/2
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),                # H
            nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, z, mask):
        m = self.mask_down(mask)  # (B, latent_dim, H/8, W/8)
        x = torch.cat([z, m], dim=1)
        return self.up(x)


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_ch=4, latent_dim=latent_dim)
        self.decoder = Decoder(mask_ch=1, latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # convert log-variance to standard deviation
        eps = torch.randn_like(std)  # random noise sampled from a standard normal distribution N(0,1)
        return mu + eps * std  # apply shift (mu) and scale (std)

    def forward(self, img, mask):
        mu, logvar = self.encoder(img, mask)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z, mask)
        return out, mu, logvar


def vae_loss_l1(recon, img, mu, logvar, beta=1.0):
    recon_loss = F.l1_loss(recon, img)  # L1 recommended for images
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


def vae_loss_l2(recon, img, mu, logvar, beta=1.0):
    bce = F.mse_loss(recon, img, reduction='mean')  # L2 recon; you can replace with L1
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld, bce, kld


def sample_and_save(model: nn.Module, masks, device, out_path: str, epoch: int, n_per_mask=4):
    model.eval()
    with torch.no_grad():
        B = masks.size(0)
        for i in range(B):
            mask = masks[i:i+1].to(device)
            for s in range(n_per_mask):
                md = model.decoder.mask_down(mask)
                z = torch.randn(1, model.latent_dim, md.shape[2], md.shape[3], device=device)
                synth = model.decoder(z, mask)
                im_path = os.path.join(out_path, f'sample_e{epoch}_mask{i}_s{s}.png')
                save_image(synth, im_path, normalize=False)


def train_loop(
        model: nn.Module,
        dataloader,
        epochs=10,
        lr=1e-4,
        loss_type=LossType.L1_LOSS,
        beta=1.0,
        out_dir=None,
        device=torch.device('cuda')
):
    out_epoch_imgs = os.path.join(out_dir, 'recon_epoch_images')
    os.makedirs(out_epoch_imgs, exist_ok=True)
    log_utils.info(f'{out_epoch_imgs=}')

    out_conditioned_path = os.path.join(out_dir, 'img_conditioned_on_masks')
    os.makedirs(out_conditioned_path, exist_ok=True)
    log_utils.info(f'{out_conditioned_path=}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc="Train cvae", position=0, leave=True)
        running_loss = 0.0
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            recon_images, mu, logvar = model(imgs, masks)

            if loss_type is LossType.L1_LOSS:
                loss, rec, kld = vae_loss_l1(recon_images, imgs, mu, logvar, beta=beta)
            elif loss_type is LossType.L2_LOSS:
                loss, rec, kld = vae_loss_l2(recon_images, imgs, mu, logvar, beta=beta)
            else:
                raise Exception('Invalid arg', loss_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch}, loss:{running_loss/ (pbar.n+1):.4f} rec:{rec:.4f} kld:{kld:.4f}")
        pbar.close()

        n_img_eval = 8
        model.eval()
        with torch.no_grad():
            sample_imgs = imgs[:n_img_eval]
            sample_masks = masks[:n_img_eval]
            recon_images, _, _ = model(sample_imgs, sample_masks)
            # save real / recon / mask triplets
            grid = torch.cat([sample_imgs, recon_images, sample_masks.repeat(1, 3, 1, 1)], dim=0)
            out_im_path = os.path.join(out_epoch_imgs, f"out_epoch_{epoch:03d}.png")
            save_image(grid, out_im_path, nrow=n_img_eval, normalize=False)
            sample_and_save(model, sample_masks.cpu(), device, out_conditioned_path, epoch=epoch, n_per_mask=3)

    return model


def train_cvae(cfg):
    img_dir = cfg.img_dir
    mask_dir = cfg.mask_dir
    out_dir = cfg.out_dir

    device = cfg.device
    img_size = cfg.img_size

    latent_dim = cfg.latent_dim
    batch_size = cfg.batch_size
    num_workers = cfg.dataloader_workers
    epochs = cfg.epochs
    lr = cfg.learning_rate
    loss_type = cfg.loss_type
    beta = cfg.beta

    im_mask_dataset = TrainImageMaskDataset(img_dir, mask_dir, img_size=img_size)
    dataloader = DataLoader(
        im_mask_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    log_utils.info('Total files in dataset', len(im_mask_dataset))

    model = ConditionalVAE(latent_dim=latent_dim).to(device)
    model = train_loop(
        model,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        beta=beta,
        out_dir=out_dir,
        device=device
    )
    ckpt_vae_out_path = os.path.join(out_dir, "cvae_mask2img.pth")
    torch.save(model.state_dict(), ckpt_vae_out_path)
    log_utils.info("Training finished")


def run_main():
    host_type = utils.detect_host_type()
    log_utils.info(f'{host_type=}')

    defect_data_type = RealDefectDatasetType.TYPE1_CROP
    img_dir, mask_dir = vae_utils.get_real_mask_image_dirs(host_type, defect_data_type)

    out_write_dir: Final[Path] = Path(config.TMP_FILES_DIR) / 'vae2'
    os.makedirs(out_write_dir, exist_ok=True)

    debug = False
    img_size = 256
    latent_dim = 128
    batch_size = 32
    num_workers = 4
    epochs = 100
    lr = 1e-4
    beta = 1.0
    loss_type = LossType.L1_LOSS

    cfg = CFG(
        debug=debug,
        host_type=host_type,

        defect_data_type=defect_data_type,
        img_dir=img_dir,
        mask_dir=mask_dir,

        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        dataloader_workers=num_workers,
        img_size=img_size,
        loss_type=loss_type,
        beta=beta,

        out_write_dir=str(out_write_dir)
    )
    shutil.rmtree(cfg.out_dir, ignore_errors=True)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    log_level = "DEBUG" if cfg.debug else "INFO"
    log_utils.init(cfg.out_dir, filename='logs.log', level=log_level, overwrite=True)

    train_cvae(cfg)


if __name__ == "__main__":
    run_main()
