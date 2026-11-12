"""Aggregate horizon AUC for checkpoint selection; no row-level export."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..config import evaluation_times


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = {key: [] for key in ('scores', 'labels', 'masks', 'categories', 'sex')}
    for batch in loader:
        outputs, _, _ = model(batch['features'].to(device), batch['times'].to(device))
        rows['scores'].append(outputs.sigmoid().permute(1, 0, 2).cpu().numpy())
        for key in rows.keys() - {'scores'}:
            rows[key].append(batch[key].numpy())
    rows = {key: np.concatenate(value) for key, value in rows.items()}
    aucs = {}
    for category in range(1, 6):
        eligible = (rows['categories'] == category) | (rows['categories'] == 0)
        if category == 2:
            eligible &= rows['sex'] == 1
        elif category == 4:
            eligible &= rows['sex'] == 0
        for col, day in enumerate(evaluation_times):
            valid = eligible & (rows['masks'][:, col] == 1)
            labels = rows['labels'][valid, col]
            key = f'category_{category}_day_{day}'
            aucs[key] = (float(roc_auc_score(labels, rows['scores'][valid, category - 1, col]))
                         if np.unique(labels).size == 2 else None)
    available = [value for value in aucs.values() if value is not None]
    return {'mean_auc': float(np.mean(available)) if available else None, 'auc': aucs}
