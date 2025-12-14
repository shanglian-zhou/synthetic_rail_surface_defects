## Dependencies and Environment

# This code requires the same dependencies and virtual environment setup as the official ControlNet implementation:
# https://github.com/lllyasviel/ControlNet

# Please follow the environment setup instructions in the ControlNet repository before running this code.


# === Disable xformers by monkey-patching ===
import torch

try:
    import xformers.ops

    def fallback_attention(q, k, v, attn_bias=None, op=None, p=0.0):
        scale = q.shape[-1] ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

    xformers.ops.memory_efficient_attention = fallback_attention
    xformers.ops.memory_efficient_attention_forward = fallback_attention

    print("⚠️ xformers disabled: using fallback PyTorch attention.")
except ImportError:
    print("✅ xformers not installed; using default PyTorch attention.")


# === Standard imports ===
from tqdm import tqdm
import os
import random
import numpy as np
import cv2
from PIL import Image
from glob import glob
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# ========== Configuration ==========
NUM_SAMPLES = 5
SEED = 42
PROMPT = "rail track surface defect with spalling"

# for sd-v1.5
SOURCE_DIR = "training/rsdds/source"
OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = "checkpoints/last.ckpt"
BATCH_SIZE = 25
file_predix = "sd_v15_"
CONFIG_PATH = "models/cldm_v15.yaml"

# for sd-v2.1
SOURCE_DIR = "test/source_20251202_sd21"
OUTPUT_DIR = "test/target_20251202_sd21"
CHECKPOINT_PATH = "checkpoints_20251202_sd21/last.ckpt"
BATCH_SIZE = 25
file_predix = ""
CONFIG_PATH = "models/cldm_v21.yaml"



file_predix = "sd_v15_"

STEPS = 50
ETA = 0.0
GUIDANCE_SCALE = 9.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

# ========== Load Model ==========
print("🔧 Loading ControlNet model...")

model = create_model(CONFIG_PATH).cpu()
model.load_state_dict(load_state_dict(CHECKPOINT_PATH, location="cpu"))
model = model.half().cuda()   # FP16 model
model.eval()

ddim_sampler = DDIMSampler(model)


# ========== Select Random Source Images ==========
all_images = sorted(glob(os.path.join(SOURCE_DIR, "*.png")))
selected_images = random.sample(all_images, NUM_SAMPLES)

# 20251202
selected_images = all_images
NUM_SAMPLES = len(selected_images)

print(f"🎯 Selected {NUM_SAMPLES} samples for inference.")


# === Group into mini-batches ===
for start_idx in range(0, len(selected_images), BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, len(selected_images))
    batch_paths = selected_images[start_idx:end_idx]
    batch_images = []
    batch_filenames = []

    for image_path in batch_paths:
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        batch_filenames.append(base_name)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_rgb, dtype=torch.float16) / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # (3, H, W)
        batch_images.append(image_tensor)

    # === Stack and Move to GPU ===
    batch_tensor = torch.stack(batch_images).cuda()
    curr_batch_size = batch_tensor.size(0)

    # === Conditioning ===
    cond = {
        "c_concat": [batch_tensor],
        "c_crossattn": [model.get_learned_conditioning([PROMPT] * curr_batch_size).half()]
    }
    uc = {
        "c_concat": [batch_tensor],
        "c_crossattn": [model.get_learned_conditioning([""] * curr_batch_size).half()]
    }

    # === Sampling ===
    latent_shape = (4, 32, 32)
    print(f"🚀 Sampling batch {start_idx} to {end_idx - 1}...")

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        samples, _ = ddim_sampler.sample(
            S=STEPS,
            conditioning=cond,
            batch_size=curr_batch_size,
            shape=latent_shape,
            verbose=False,
            unconditional_guidance_scale=GUIDANCE_SCALE,
            unconditional_conditioning=uc,
            eta=ETA,
        )
        decoded = model.decode_first_stage(samples)

    # === Save outputs ===
    decoded = ((decoded.clamp(-1, 1) + 1) / 2.0).cpu().numpy()
    for i in range(curr_batch_size):
        img = (decoded[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        save_path = os.path.join(OUTPUT_DIR, f"{file_predix}{start_idx + i + 1}_{batch_filenames[i]}.png")
        Image.fromarray(img).save(save_path)
        print(f"✅ Saved: {save_path}")
