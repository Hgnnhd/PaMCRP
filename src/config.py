"""Model settings only; no source-data or cohort configuration."""

evaluation_times = [90, 180, 360, 1080, 1800]  # days; output column order
NUM_CATEGORIES = 6  # reference category plus five cancer categories
DIMS = 256
MAX_SEQ_LENGTH = 241
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 16
