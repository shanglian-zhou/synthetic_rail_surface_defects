## Dependencies and Environment

# This code requires the same dependencies and virtual environment setup as the official ControlNet implementation:
# https://github.com/lllyasviel/ControlNet

# Please follow the environment setup instructions in the ControlNet repository before running this code.

# === Disable xformers by monkey-patching ===
import torch

try:
    import xformers.ops


    def fallback_attention(q, k, v, attn_bias=None, op=None, p=0.0):
        # Simple scaled dot-product attention fallback
        scale = q.shape[-1] ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


    # Monkey-patch the xformers memory-efficient attention ops
    xformers.ops.memory_efficient_attention = fallback_attention
    xformers.ops.memory_efficient_attention_forward = fallback_attention

    print("⚠️ xformers disabled: using fallback PyTorch attention.")

except ImportError:
    print("✅ xformers not installed; using default PyTorch attention.")


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_20251201 import MyDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
num_workers = 0
precision = 32
sd_locked = True
max_epochs = 10

batch_size = 60
num_workers = 48
precision = 16
sd_locked = False
max_epochs = 100
logger_freq = 300
learning_rate = 1e-4
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints_20251202",
    filename="controlnet-best-{epoch:02d}-{train_loss:.4f}",
    monitor="train/loss_epoch",  # metric to monitor
    mode="min",  # lower = better
    save_top_k=3,  # keep only the best checkpoint
    save_last=True,  # also save last.ckpt
)

trainer = pl.Trainer(gpus=1,
                     precision=precision,
                     max_epochs=max_epochs,
                     callbacks=[logger, checkpoint_callback])

# Train!
trainer.fit(model, dataloader)
