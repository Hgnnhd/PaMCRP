from torch.utils.data import Dataset, DataLoader
import torch
import random
import pickle
import numpy as np
import os
import json
from datetime import datetime
from config import *
from sklearn.utils import check_random_state
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
import torch.nn as nn
import torch.nn.functional as F

def orthogonality_promoting_regularization(W):
    if W.dim() > 2:
        W = W.view(W.size(0), -1)
    WtW = torch.matmul(W.t(), W)
    return torch.norm(WtW * (1 - torch.eye(WtW.shape[0]).to(W.device)), p='fro')

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, no_norm=False, use_attention=False, gamma=2):
        super(OrthogonalProjectionLoss, self).__init__()
        self.weights_dict = None
        self.no_norm = no_norm
        self.gamma = gamma
        self.use_attention = use_attention

    def forward(self, features, labels=None):
        device = (features.device if features.is_cuda else torch.device('cpu'))
        if self.use_attention:
            features_weights = torch.matmul(features, features.T)
            features_weights = F.softmax(features_weights, dim=1)
            features = torch.matmul(features_weights, features)
        #  features are normalized
        if not self.no_norm:
            features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim
        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (self.gamma * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


class OrthogonalityLosses:
    @staticmethod
    def basic_orthogonal_loss(W):
        """
         W^T * W 
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        WtW = torch.matmul(W.t(), W)
        identity = torch.eye(WtW.size(0), device=W.device)
        return torch.norm(WtW - identity)

    @staticmethod
    def flexible_orthogonal_loss(W, row_wise=False):
        """
        
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        if row_wise:
            WWt = torch.matmul(W, W.t())
            identity = torch.eye(WWt.size(0), device=W.device)
            return torch.norm(WWt - identity)
        else:
            WtW = torch.matmul(W.t(), W)
            identity = torch.eye(WtW.size(0), device=W.device)
            return torch.norm(WtW - identity)

    @staticmethod
    def off_diagonal_orthogonal_loss(W):
        """
        
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        WtW = torch.matmul(W.t(), W)
        mask = 1 - torch.eye(WtW.shape[0], device=W.device)
        return torch.norm(WtW * mask, p='fro')

    @staticmethod
    def soft_orthogonality_constraint(W, lambda_val=1.0):
        """
         log-determinant 
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        WtW = torch.matmul(W.t(), W)
        return -torch.logdet(WtW + lambda_val * torch.eye(WtW.shape[0], device=W.device))

    @staticmethod
    def spectral_restricted_orthogonality(W, k=1):
        """
        
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        WtW = torch.matmul(W.t(), W)
        eigenvalues = torch.linalg.eigvalsh(WtW)
        top_k = torch.topk(eigenvalues, k).values
        return torch.sum(top_k) - k

    @staticmethod
    def cosine_orthogonality_loss(W):
        """
        
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        W_normalized = F.normalize(W, p=2, dim=1)
        cosine_sim = torch.matmul(W_normalized, W_normalized.t())
        return torch.norm(cosine_sim - torch.eye(cosine_sim.shape[0], device=W.device))

    @staticmethod
    def double_soft_orthogonality_constraint(W, lambda_val=1.0):
        """
         W^T * W  W * W^T
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        WtW = torch.matmul(W.t(), W)
        WWt = torch.matmul(W, W.t())
        loss_WtW = -torch.logdet(WtW + lambda_val * torch.eye(WtW.shape[0], device=W.device))
        loss_WWt = -torch.logdet(WWt + lambda_val * torch.eye(WWt.shape[0], device=W.device))
        return loss_WtW + loss_WWt




@torch.jit.script
def build_hierarchical_adjacency_matrices(data: torch.Tensor, num_nodes: int, num_levels: int) -> torch.Tensor:
    adj_matrices = []
    eye = torch.eye(num_nodes, device=data.device)
    for level in range(num_levels - 1):
        level_mapping = data[level]
        same_group = (level_mapping.unsqueeze(1) == level_mapping.unsqueeze(0)).float()
        adj_matrices.append(same_group)
    return torch.stack(adj_matrices)

@torch.jit.script
def compute_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    if len(adj.shape) == 3:
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        D_inv_sqrt = D_inv_sqrt.unsqueeze(-1)
        D_adj = D_inv_sqrt * adj * D_inv_sqrt.transpose(-1, -2)
        L = torch.eye(adj.size(-1), device=adj.device) - D_adj
        return L
    else:
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        D_inv_sqrt = D_inv_sqrt.unsqueeze(-1)
        D_adj = D_inv_sqrt * adj * D_inv_sqrt.transpose(-1, -2)
        return D_adj

@torch.jit.script
def precompute_laplacians_A(data: torch.Tensor, num_nodes: int, num_levels: int) -> torch.Tensor:
    all_L = []
    for i in range(data.size(0)):
        A = build_hierarchical_adjacency_matrices(data[i], num_nodes, num_levels)
        L = compute_normalized_laplacian(A)
        all_L.append(L)
    return torch.stack(all_L)
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # [batch_size, num_nodes, out_features]
        e = self._prepare_attentional_mechanism_input(Wh)  # [batch_size, num_nodes, num_nodes]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)
        return F.leaky_relu(e.squeeze(-1), negative_slope=self.alpha)
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([GATLayer(nfeat, nhid//nheads, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.out_att = GATLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        return x



def print_results(results, metrics_to_display=None):
    output = []
    time_points = list(set([key.split('_')[1] for key in results.keys() if key.startswith('Time_')]))
    time_points = sorted(time_points, key=int)

    category_names = {
        1: "Lung",
        2: "Breast",
        3: "Colorectal",
        4: "Prostate",
        5: "Stomach",
    }

    # Print AUC results
    output.append("\nCategory-specific AUC Results:")
    categories = range(1, 6)
    header = f"{'Category':<10}"
    for time_point in time_points:
        header += f"{time_point:^28}"
    header += f"{'Average':^12}"
    output.append(header)
    output.append("-" * (10 + 28 * len(time_points) + 12))

    for category in categories:
        classes = category_names[category]
        line = f"{classes:<10}"
        avg_auc = []
        for time_point in time_points:
            key = f"Time_{time_point}_{category}_AUC"
            value = results.get(key, "N/A")
            if isinstance(value, float):
                ci_lower, ci_upper = results.get(f"Time_{time_point}_{category}_AUC_CI", (None, None))
                line += f"{value:.4f} ({ci_lower:.4f}-{ci_upper:.4f})".center(28)
                avg_auc.append(value)
            else:
                line += f"{value:^28}"

        if avg_auc:
            avg = np.mean(avg_auc)
            line += f"{avg:.4f}".center(12)
        else:
            line += f"{'N/A':^12}"
        output.append(line)

    # Print C-index results
    output.append("\nCategory-specific C-index Results:")
    header = f"{'Category':<10}{'C-index':^12}{'CI':^24}"
    output.append(header)
    output.append("-" * 46)

    for category in categories:
        classes = category_names[category]
        c_index = results.get(f"C_index_{category}", "N/A")
        ci_lower, ci_upper = results.get(f"C_index_CI_{category}", ("N/A", "N/A"))

        line = f"{classes:<10}"
        if isinstance(c_index, float):
            line += f"{c_index:.4f}".center(12)
        else:
            line += f"{c_index:^12}"

        if isinstance(ci_lower, float) and isinstance(ci_upper, float):
            line += f"({ci_lower:.4f}-{ci_upper:.4f})".center(24)
        else:
            line += f"({ci_lower}-{ci_upper})".center(24)

        output.append(line)

    return "\n".join(output)


def save_training_results(results, name, filename_prefix, model_name=None):

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    results_filename = f'{filename_prefix}_{current_time}_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{model_name}.json'


    printed_results = print_results(results)


    print(printed_results)

    txt_filename = f'{filename_prefix}_{current_time}_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{model_name}_printed_results.txt'
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(printed_results)
        txt_file.write("\n")
        txt_file.write(name)

    return results_filename, txt_filename

# def collate_fn(batch):

#     data, aux, label, masks, time, gold, top5_gold = zip(*batch)

#     max_len = max(d.shape[1] for d in data)

#     padded_data = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in data]
#     padded_time = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in time]
#

#     data = torch.stack(padded_data)
#     time = torch.stack(padded_time)
#     aux = torch.stack(aux)
#     label = torch.stack(label)
#     masks = torch.stack(masks)
#     gold = torch.stack(gold)
#     top5_gold = torch.stack(top5_gold)
#
#     return data, aux, label, masks, time, gold, top5_gold

def adaptive_structured_sampling(data, num_timesteps_to_select):
    """
    
    """
    T = data.shape[2]


    changes = torch.abs(data[:, :, 1:].float() - data[:, :, :-1].float()).mean(dim=(0, 1))


    change_cumsum = torch.cumsum(changes, 0)
    total_change = change_cumsum[-1]


    target_changes = torch.linspace(0, total_change, num_timesteps_to_select + 1)[1:].to(data.device)
    selected_indices = []

    for target in target_changes:
        idx = torch.searchsorted(change_cumsum, target)
        selected_indices.append(idx.item())

    return torch.tensor(selected_indices)


def collate_fn(batch):

    data, aux, label, masks, time, gold, top5_gold = zip(*batch)


    max_len = max(d.shape[1] for d in data)


    padded_data = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in data]
    padded_time = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in time]


    data = torch.stack(padded_data)  # (B, H, T)
    time = torch.stack(padded_time)  # (B, H, T)
    aux = torch.stack(aux)
    label = torch.stack(label)
    masks = torch.stack(masks)
    gold = torch.stack(gold)
    top5_gold = torch.stack(top5_gold)

    return data, aux, label, masks, time, gold, top5_gold


def collate_fn_train(batch, feature_fraction=0.5):

    data, aux, label, masks, time, gold, top5_gold = zip(*batch)

    max_len = max(d.shape[1] for d in data)

    padded_data = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in data]
    padded_time = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in time]

    data = torch.stack(padded_data)  # (B, H, T)
    time = torch.stack(padded_time)  # (B, H, T)
    aux = torch.stack(aux)
    label = torch.stack(label)
    masks = torch.stack(masks)
    gold = torch.stack(gold)
    top5_gold = torch.stack(top5_gold)

    B, H, T = data.shape
    num_timesteps_to_select = max(1, int(feature_fraction * T))
    stride = max(1, T // num_timesteps_to_select)
    start_idx = torch.randint(0, stride, (1,)).item()
    selected_indices = torch.arange(start_idx, T, stride)[:num_timesteps_to_select]

    selected_indices = torch.sort(selected_indices)[0]

    sampled_data = data[:, :, selected_indices]
    sampled_time = time[:, :, selected_indices]

    if masks.dim() == 3 and masks.shape[2] == T:
        masks = masks[:, :, selected_indices]

    return sampled_data, aux, label, masks, sampled_time, gold, top5_gold


def create_balanced_sampler(dataset):

    targets = dataset.top5_gold
    targets = torch.tensor(targets)

    class_sample_count = torch.bincount(targets)

    weight = 1. / class_sample_count.float()
    samples_weight = weight[targets]

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler

class BalancedShuffleSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indices = self.create_balanced_indices()

    def create_balanced_indices(self):

        labels = self.dataset.top5_gold
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        return list(torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight)))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.dataset)



class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = torch.Tensor(self.data[idx]).to(device)
        label = np.array(self.label[idx], dtype=np.int64)
        label = torch.from_numpy(label).to(device)

        return data, label


class CustomDataset_aux(Dataset):

    def __init__(self, data, aux, label, masks, time, gold, top5_gold=None, mode=None):
        self.data = data
        self.aux = aux
        self.label = label
        self.masks = masks
        self.gold = gold
        self.mode = mode
        self.time = time
        if top5_gold is None:
            self.top5_gold = self.gold
        else:
            self.top5_gold = top5_gold

        self.random_state = 42


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        random_state = check_random_state(self.random_state)

        original_data = self.data[idx]
        original_data = torch.Tensor(original_data).to(device)
        data = original_data.long()

        aux = torch.Tensor(self.aux[idx]).to(device)
        label = np.array(self.label[idx], dtype=np.int64)
        label = torch.from_numpy(label).to(device)
        masks = torch.Tensor(self.masks[idx]).to(device)
        gold = torch.tensor(self.gold[idx]).to(device)
        time = torch.tensor(self.time[idx]).to(device)
        top5_gold = torch.tensor(self.top5_gold[idx]).to(device)

        return data, aux, label, masks, time, gold, top5_gold






def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def save_model_architecture(model, filename_prefix):

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    architecture_filename = f'{filename_prefix}_{current_time}.json'


    model_architecture = {}

    with open(architecture_filename, 'w') as json_file:
        json.dump(model_architecture, json_file)

    return architecture_filename




def save_model(model, filename_prefix):

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_filename = f'{filename_prefix}_{current_time}.pth'

    torch.save(model.state_dict(), model_filename)

    return model_filename



class CustomRandomUnderSampler:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, features, labels):
        random_state = check_random_state(self.random_state)


        unique_classes, class_counts = np.unique(labels, return_counts=True)


        target_count = np.min(class_counts)


        resampled_features = []
        resampled_adj = []
        resampled_labels = []

        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            sampled_indices = random_state.choice(class_indices, target_count, replace=False)

            resampled_features.extend([features[i] for i in sampled_indices])
            # resampled_adj.extend([adj[i] for i in sampled_indices])
            resampled_labels.extend([labels[i] for i in sampled_indices])


        shuffle_indices = random_state.permutation(len(resampled_labels))
        resampled_features = [resampled_features[i] for i in shuffle_indices]
        # resampled_adj = [resampled_adj[i] for i in shuffle_indices]
        resampled_labels = [resampled_labels[i] for i in shuffle_indices]

        return resampled_features, resampled_labels

class CustomRandomUnderSampler_aux:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, features, aux, label, labels, masks):
        random_state = check_random_state(self.random_state)

        targets = [np.argmax(label) + 1 if np.sum(label) > 0 else 0 for label in labels]
        targets = np.array(targets)



        unique_classes, class_counts = np.unique(targets, return_counts=True)


        target_count = np.min(class_counts)


        resampled_features = []
        resampled_aux = []
        resampled_labels = []
        resampled_masks = []
        resampled_label = []

        for class_label in unique_classes:
            class_indices = np.where(targets == class_label)[0]
            sampled_indices = random_state.choice(class_indices, target_count, replace=False)

            resampled_features.extend([features[i] for i in sampled_indices])
            resampled_aux.extend([aux[i] for i in sampled_indices])
            resampled_labels.extend([labels[i] for i in sampled_indices])
            resampled_masks.extend([masks[i] for i in sampled_indices])
            resampled_label.extend([label[i] for i in sampled_indices])


        shuffle_indices = random_state.permutation(len(resampled_labels))
        resampled_features = [resampled_features[i] for i in shuffle_indices]
        resampled_aux = [resampled_aux[i] for i in shuffle_indices]
        resampled_labels = [resampled_labels[i] for i in shuffle_indices]
        resampled_masks = [resampled_masks[i] for i in shuffle_indices]
        resampled_label = [resampled_label[i] for i in shuffle_indices]

        return resampled_features, resampled_aux, resampled_labels, resampled_masks, resampled_label


class CustomRandomOverSampler_aux:
    def __init__(self, random_state=None, sampling_method='sliding_window'):
        self.random_state = random_state
        self.sampling_method = sampling_method

    def sample_custom_fraction_features(self, feature, random_state, feature_fraction=0.7):
        feature = np.array(feature)
        feature_size = feature.shape[1]

        num_features_to_select = max(1, int(feature_fraction * feature_size))

        selected_indices = random_state.choice(feature_size, size=num_features_to_select, replace=False)

        selected_indices.sort()

        return feature[:, selected_indices]

    def sample_features(self, feature, random_state):
        feature = np.array(feature)
        feature_size = feature.shape[1]
        start = random_state.randint(0, feature_size)
        end = random_state.randint(start + 1, feature_size + 1)
        return feature[:,start:end]

    def fit_resample(self, features, aux, labels, masks, time, gold, top5_gold):
        random_state = check_random_state(self.random_state)
        unique_classes, class_counts = np.unique(top5_gold, return_counts=True)
        # target_count = np.max(class_counts)


        sorted_counts = np.sort(class_counts)[::-1]
        target_count = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]

        resampled_features = []
        resampled_aux = []
        resampled_labels = []
        resampled_masks = []
        resampled_time = []
        resampled_gold = []
        resampled_top5_gold = []

        total_samples_to_add = sum(max(0, target_count - count) for count in class_counts)

        with tqdm(total=total_samples_to_add, desc="Oversampling progress") as pbar:
            for class_label in unique_classes:
                class_indices = np.where(top5_gold == class_label)[0]
                if len(class_indices) < target_count:
                    samples_to_add = target_count - len(class_indices)
                    for _ in range(samples_to_add):
                        sample_idx = random_state.choice(class_indices)
                        # new_feature = self.sample_features(features[sample_idx], random_state)
                        new_feature = self.sample_custom_fraction_features(features[sample_idx], random_state, 0.7)
                        resampled_features.append(new_feature)
                        resampled_aux.append(aux[sample_idx])
                        resampled_labels.append(labels[sample_idx])
                        resampled_masks.append(masks[sample_idx])
                        resampled_time.append(time[sample_idx])
                        resampled_gold.append(gold[sample_idx])
                        resampled_top5_gold.append(top5_gold[sample_idx])
                        pbar.update(1)
                resampled_features.extend([features[i] for i in class_indices])
                resampled_aux.extend([aux[i] for i in class_indices])
                resampled_labels.extend([labels[i] for i in class_indices])
                resampled_masks.extend([masks[i] for i in class_indices])
                resampled_time.extend([time[i] for i in class_indices])
                resampled_gold.extend([gold[i] for i in class_indices])
                resampled_top5_gold.extend([top5_gold[i] for i in class_indices])

        return (resampled_features, resampled_aux, resampled_labels, resampled_masks,
                resampled_time, resampled_gold, resampled_top5_gold)


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
