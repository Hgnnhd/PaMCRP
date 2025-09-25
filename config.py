"""
PaMCRP configuration (edit these placeholders)

Set the values below to suit your environment and experiment. Examples are
given in comments. There are no active defaults - placeholders are None until
you set them.
"""

import torch

# Device selection
gpu_id = None  # e.g., 0 (GPU index). Leave None to fall back to CPU.

# Time horizons in days (e.g., [90, 180, 360, 1080, 1800])
evaluation_times = None  # set a list of integers

# ICD granularity level (e.g., 4)
LEVEL = None

# Exclusion windows in days
EXCLUDE_DAY = None  # modify to the days you want to exclude (e.g., 90)
EXCLUDE_DAY_Normal = None  # for normal patients (e.g., 90)

# Date format string for any file-level date parsing you add (if needed)
DATE_FORMAT = None  # e.g., "%Y-%m-%d"

# Dataset directory name (relative to repo root parent)
DATASET_DIR = None  # e.g., "dataset"

# Minimum number of diagnoses required per patient (e.g., 5)
Min_length = None
# Model variant flag (e.g., "ours")
model_type = None
# Embedding dimension (e.g., 256)
DIMS = None
# Target cancer patterns: list of ICD-10 "C" two-digit prefixes
# e.g., ["34","50","18","19","20","61","16"]
TARGET_PATTERN = None

# Safe device resolution (uses gpu_id when provided and CUDA is available)
if torch.cuda.is_available() and gpu_id is not None:
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device("cpu")
