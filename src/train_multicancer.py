"""Train PaMCRP on privately prepared tensors. No source-data processing."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import (DIMS, MAX_SEQ_LENGTH, NUM_ENCODER_LAYERS, NUM_HEADS,
                     evaluation_times)
from .module.data import PreparedDataset
from .module.eval import evaluate
from .module.model import MGDA
from .module.net import PaMCRP


def task_losses(model, batch):
    outputs, class_logits, proxy_loss = model(
        batch['features'], batch['times'], batch['categories'])
    risk_loss = outputs.sum() * 0.0
    for i in range(5):
        eligible = (batch['categories'] == i + 1) | (batch['categories'] == 0)
        if i == 1:
            eligible &= batch['sex'] == 1
        elif i == 3:
            eligible &= batch['sex'] == 0
        masks = batch['masks'][eligible]
        if masks.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(
                outputs[i][eligible], batch['labels'][eligible], reduction='none')
            risk_loss = risk_loss + (loss * masks).sum() / masks.sum() / 5
    classification_loss = F.cross_entropy(class_logits, batch['categories'])
    return risk_loss, classification_loss, proxy_loss


def train_epoch(model, loader, optimizer, balancer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        losses = task_losses(model, batch)
        balancer.backward(losses)
        optimizer.step()
        total += sum(loss.detach().item() for loss in losses) * len(batch['sex'])
    return total / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Private directory containing train.npz, validation.npz, test.npz')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/pamcrp'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, patience, and lr must be positive')
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    datasets = {split: PreparedDataset(args.data_dir / f'{split}.npz')
                for split in ('train', 'validation', 'test')}
    if len({dataset.sequence_length for dataset in datasets.values()}) != 1:
        raise ValueError('All splits must use the same sequence length; see DATA_FORMAT.md')
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=(split == 'train'))
               for split, dataset in datasets.items()}
    n_times = len(evaluation_times)
    model = PaMCRP(DIMS, DIMS, n_times, NUM_HEADS, MAX_SEQ_LENGTH,
                   NUM_ENCODER_LAYERS, n_times, 'ours').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    balancer = MGDA(model, task_num=3, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / 'best_model.pt'
    best_auc, stale = -float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders['train'], optimizer, balancer, device)
        metrics = evaluate(model, loaders['validation'], device)
        score = metrics['mean_auc']
        if score is None:
            raise ValueError('Validation AUC is undefined: no eligible horizon has both classes')
        print(f'Epoch {epoch}: loss={loss:.6f}, validation_mean_auc={score:.6f}')
        if score > best_auc:
            best_auc, stale = score, 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'validation_mean_auc': score, 'evaluation_times': evaluation_times},
                       checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    result = {'best_epoch': state['epoch'], 'validation_mean_auc': best_auc,
              'test': evaluate(model, loaders['test'], device)}
    (args.output_dir / 'metrics.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Training complete. Local outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
