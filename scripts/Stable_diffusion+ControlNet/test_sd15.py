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
SOURCE_DIR = "training/rsdds/source"
OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = "checkpoints/last.ckpt"

SOURCE_DIR = "test/source_20251202"
OUTPUT_DIR = "test/target_20251202"
CHECKPOINT_PATH = "checkpoints_20251202/last.ckpt"

CONFIG_PATH = "models/cldm_v15.yaml"

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


# ========== Inference Loop ==========
for idx, image_path in enumerate(selected_images):
    file_name = os.path.basename(image_path)
    base_name = os.path.splitext(file_name)[0]

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to FP16 tensor
    image_tensor = torch.tensor(image_rgb, dtype=torch.float16) / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).cuda()  # (1, 3, H, W)

    # Text embeddings in FP16
    cond = {
        "c_concat": [image_tensor],
        "c_crossattn": [model.get_learned_conditioning([PROMPT]).half()]
    }

    uc = {
        "c_concat": [image_tensor],
        "c_crossattn": [model.get_learned_conditioning([""]).half()]
    }

    # Latent shape for 256×256 = 4 × 32 × 32
    latent_shape = (4, 32, 32)

    print(f"🚀 Sampling for {file_name}...")

    # === FP16 sampling inside autocast ===
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):

        samples, _ = ddim_sampler.sample(
            S=STEPS,
            conditioning=cond,
            batch_size=1,
            shape=latent_shape,
            verbose=False,
            unconditional_guidance_scale=GUIDANCE_SCALE,
            unconditional_conditioning=uc,
            eta=ETA,
        )

        decoded = model.decode_first_stage(samples)

    # Convert to uint8 image
    decoded = ((decoded.clamp(-1, 1) + 1) / 2.0).cpu().numpy()
    decoded_img = (decoded[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    save_path = os.path.join(OUTPUT_DIR, f"{file_predix}_{idx+1}_{base_name}.png")
    Image.fromarray(decoded_img).save(save_path)

    print(f"✅ Saved output → {save_path}")

print("🎉 Inference complete.")
