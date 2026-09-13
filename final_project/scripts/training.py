import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from google.colab import drive

drive.mount("/content/drive")

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/01_speech_processing/sp_exam_project/dataset"
)
DATASET_DIR = DRIVE_ROOT / "llava_dataset"
IMAGE_DIR = DRIVE_ROOT / "images"
OUTPUT_DIR = DRIVE_ROOT / "llava_checkpoints"

TRAIN_FILE = DATASET_DIR / "llava_train.json"

MODEL_NAME_OR_PATH = "liuhaotian/llava-v1.5-7b"
LLAVA_VERSION = "v1"
VISION_TOWER = "openai/clip-vit-large-patch14-336"

LORA_ENABLE = True
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05

NUM_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
MODEL_MAX_LENGTH = 2048

SAVE_STEPS = 500
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2
NUM_WORKERS = 4

# Set to True only when using DeepSpeed configuration.
USE_DEEPSPEED = False
DEEPSPEED_CONFIG = "./scripts/zero2.json"

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required for LLaVA fine-tuning.")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

try:
    with TRAIN_FILE.open("r", encoding="utf-8") as file:
        training_records = json.load(file)
except json.JSONDecodeError as error:
    raise ValueError(f"Invalid training JSON: {TRAIN_FILE}") from error

if not isinstance(training_records, list) or not training_records:
    raise ValueError("The training JSON must contain at least one record.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("GPU:", torch.cuda.get_device_name(0))
print("Training records:", f"{len(training_records):,}")
print("Training file:", TRAIN_FILE)
print("Image folder:", IMAGE_DIR)
print("Checkpoint folder:", OUTPUT_DIR)

repo_root = Path.cwd()
training_script = repo_root / "llava" / "train" / "train_mem.py"
if not training_script.exists():
    raise FileNotFoundError(
        "Run this script from the official LLaVA repository root. "
        f"Expected: {training_script}"
    )

command = [
    sys.executable,
    str(training_script),
    "--model_name_or_path", MODEL_NAME_OR_PATH,
    "--version", LLAVA_VERSION,
    "--data_path", str(TRAIN_FILE),
    "--image_folder", str(IMAGE_DIR),
    "--vision_tower", VISION_TOWER,
    "--mm_projector_type", "mlp2x_gelu",
    "--mm_vision_select_layer", "-2",
    "--mm_use_im_start_end", "False",
    "--mm_use_im_patch_token", "False",
    "--image_aspect_ratio", "pad",
    "--group_by_modality_length", "True",
    "--bf16", "True",
    "--output_dir", str(OUTPUT_DIR),
    "--num_train_epochs", str(NUM_EPOCHS),
    "--per_device_train_batch_size", str(PER_DEVICE_TRAIN_BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
    "--learning_rate", str(LEARNING_RATE),
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", str(LOGGING_STEPS),
    "--save_strategy", "steps",
    "--save_steps", str(SAVE_STEPS),
    "--save_total_limit", str(SAVE_TOTAL_LIMIT),
    "--model_max_length", str(MODEL_MAX_LENGTH),
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", str(NUM_WORKERS),
    "--lazy_preprocess", "True",
    "--report_to", "none",
]

if LORA_ENABLE:
    command.extend([
        "--lora_enable", "True",
        "--lora_r", str(LORA_R),
        "--lora_alpha", str(LORA_ALPHA),
        "--lora_dropout", str(LORA_DROPOUT),
    ])

if USE_DEEPSPEED:
    command[1:1] = ["--deepspeed", DEEPSPEED_CONFIG]

print("\nStarting LLaVA fine-tuning...")
print(" ".join(str(part) for part in command))
subprocess.run(command, check=True)

print("\nLLaVA fine-tuning completed.")
print(f"Checkpoints saved in: {OUTPUT_DIR}")
