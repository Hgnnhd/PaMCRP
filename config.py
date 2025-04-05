import torch
gpu_id = 1
evaluation_times = [
    30*3,
    30*6,
    30*12,
    30*12*3, 30*12*5]
LEVEL = 4
EXCLUDE_DAY = 1
EXCLUDE_DAY_Normal = 30*3
DATE_FORMAT = "%Y-%m-%d"
DATASET_DIR = "dataset"
Min_length = 5
model_type = "ours"
DIMS = 256
if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(gpu_id))
else:
    device = torch.device("cpu")

