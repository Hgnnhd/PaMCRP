"""Load and validate prepared tensors without deriving any data fields."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import MAX_SEQ_LENGTH, evaluation_times


class PreparedDataset(Dataset):
    KEYS = {'features', 'times', 'labels', 'masks', 'categories', 'sex'}

    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing prepared tensor file: {path}. See DATA_FORMAT.md')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != self.KEYS:
                raise ValueError('NPZ fields must be exactly: ' + ', '.join(sorted(self.KEYS)))
            arrays = {key: archive[key] for key in self.KEYS}
        x = arrays['features']
        if x.ndim != 3 or x.shape[1] != 4 or not 1 <= x.shape[2] <= MAX_SEQ_LENGTH:
            raise ValueError('features must have shape [N, 4, L], with 1 <= L <= 241')
        n, _, length = x.shape
        if n == 0:
            raise ValueError('Empty splits are not supported')
        shapes = {'times': (n, 3, length), 'labels': (n, len(evaluation_times)),
                  'masks': (n, len(evaluation_times)), 'categories': (n,), 'sex': (n,)}
        for key, value in arrays.items():
            if key in shapes and value.shape != shapes[key]:
                raise ValueError(f'{key} must have shape {shapes[key]}')
            if value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
                raise ValueError(f'{key} must contain finite numeric values')
        for key, high in [('features', 9998), ('categories', 5), ('sex', 1)]:
            value = arrays[key]
            if value.dtype.kind not in 'iu' or (value < 0).any() or (value > high).any():
                raise ValueError(f'{key} must contain integers in [0, {high}]')
        for key in ('labels', 'masks'):
            if not np.isin(arrays[key], [0, 1]).all():
                raise ValueError(f'{key} must be binary')
        self.sequence_length = length
        self.tensors = {
            key: torch.tensor(value, dtype=torch.long if key in
                              ('features', 'categories', 'sex') else torch.float32)
            for key, value in arrays.items()}
        if any(not torch.isfinite(value).all() for value in self.tensors.values()):
            raise ValueError('Values must remain finite in the model tensor dtype')

    def __len__(self):
        return len(self.tensors['features'])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.tensors.items()}
