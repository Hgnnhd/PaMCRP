import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.dim_per_head = hidden_dim // num_heads
        self.aggregate_fc = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, x, prompt=None):
        B, N, H = x.size()

        k = self.key(x).view(B, N, self.num_heads, self.dim_per_head)
        q = self.query(x).view(B, N, self.num_heads, self.dim_per_head)
        v = self.value(x).view(B, N, self.num_heads, self.dim_per_head)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        h = self.attention(q, k, v)
        h = h.transpose(1, 2).contiguous().view(B, -1, H)
        output = self.aggregate_fc(h)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, nhead):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(input_dim, nhead)
        self.layernorm_attn = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.layernorm_fc = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        h = self.multihead_attention(x)
        x = self.layernorm_attn(h + x)
        h = self.fc2(self.relu(self.fc1(x)))
        x = self.layernorm_fc(h + x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, max_seq_length, num_encoder_layers):
        super(Transformer, self).__init__()

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(input_dim, nhead)
            for _ in range(num_encoder_layers)
        ])

        self.fc_out = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.fc_out(x)
        return x


class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(CumulativeProbabilityLayer, self).__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=False)
        self.prompt_fc = nn.Linear(num_features, max_followup)
        self.age_fc = nn.Linear(1, max_followup)
        self.fusion_fc = nn.Linear(max_followup * 2, max_followup)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triagular_mask', mask)

    def hazards(self, x, prompt=None, age=None):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x, prompt=None):
        hazards = self.hazards(x, prompt)
        B, T = hazards.size()
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob


class MGDA(nn.Module):
    def __init__(self, model, task_num=6, device='cuda'):
        super(MGDA, self).__init__()
        self.model = model
        self.device = device
        self.task_num = task_num
        self._compute_grad_dim()

    def _compute_grad_dim(self):
        self.grad_dim = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _compute_grad(self, losses):
        grads = torch.zeros(self.task_num, self.grad_dim, device=self.device)
        for i, loss in enumerate(losses):
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                              for p in self.model.parameters() if p.requires_grad])
            grads[i] = grad
        return grads

    def _find_min_norm_element(self, grads):

        def _min_norm_element_from2(v1v1, v1v2, v2v2):
            if v1v2 >= v1v1:
                gamma = 0.999
                cost = v1v1
                return gamma, cost
            if v1v2 >= v2v2:
                gamma = 0.001
                cost = v2v2
                return gamma, cost
            gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
            cost = v2v2 + gamma * (v1v2 - v2v2)
            return gamma, cost

        def _min_norm_2d(grad_mat):
            dmin = float('inf')
            for i in range(grad_mat.size()[0]):
                for j in range(i + 1, grad_mat.size()[0]):
                    c, d = _min_norm_element_from2(grad_mat[i, i], grad_mat[i, j], grad_mat[j, j])
                    if d < dmin:
                        dmin = d
                        sol = [(i, j), c, d]
            return sol

        def _projection2simplex(y):
            m = len(y)
            sorted_y = torch.sort(y, descending=True)[0]
            tmpsum = 0.0
            tmax_f = (torch.sum(y) - 1.0) / m
            for i in range(m - 1):
                tmpsum += sorted_y[i]
                tmax = (tmpsum - 1) / (i + 1.0)
                if tmax > sorted_y[i + 1]:
                    tmax_f = tmax
                    break
            return torch.max(y - tmax_f, torch.zeros(m).to(y.device))

        def _next_point(cur_val, grad, n):
            proj_grad = grad - (torch.sum(grad) / n)
            tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
            tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

            torch.sum(tm1 < 1e-7) + torch.sum(tm2 < 1e-7)
            t = torch.ones(1).to(grad.device)
            if (tm1 > 1e-7).sum() > 0:
                t = torch.min(tm1[tm1 > 1e-7])
            if (tm2 > 1e-7).sum() > 0:
                t = torch.min(t, torch.min(tm2[tm2 > 1e-7]))

            next_point = proj_grad * t + cur_val
            next_point = _projection2simplex(next_point)
            return next_point

        MAX_ITER = 250
        STOP_CRIT = 1e-5

        grad_mat = grads.mm(grads.t())
        init_sol = _min_norm_2d(grad_mat)

        n = grads.size()[0]
        sol_vec = torch.zeros(n).to(grads.device)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec

        iter_count = 0

        while iter_count < MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = _next_point(sol_vec, grad_dir, n)

            v1v1 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n) * sol_vec.unsqueeze(0).repeat(n, 1) * grad_mat)
            v1v2 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n) * new_point.unsqueeze(0).repeat(n, 1) * grad_mat)
            v2v2 = torch.sum(new_point.unsqueeze(1).repeat(1, n) * new_point.unsqueeze(0).repeat(n, 1) * grad_mat)

            nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < STOP_CRIT:
                return sol_vec
            sol_vec = new_sol_vec
            iter_count += 1
        return sol_vec

    def _gradient_normalizers(self, grads, loss_data, ntype='l2'):
        if ntype == 'l2':
            gn = grads.pow(2).sum(-1).sqrt()
        elif ntype == 'loss':
            gn = loss_data
        elif ntype == 'loss+':
            gn = loss_data * grads.pow(2).sum(-1).sqrt()
        elif ntype == 'none':
            gn = torch.ones_like(loss_data).to(self.device)
        else:
            raise ValueError('No support normalization type {} for MGDA'.format(ntype))
        grads = grads / gn.clamp_min(1e-12).unsqueeze(1)
        return grads

    def backward(self, losses):
        grads = self._compute_grad(losses)
        loss_data = torch.tensor([loss.item() for loss in losses]).to(self.device)
        grads = self._gradient_normalizers(grads, loss_data, ntype="l2")
        sol = self._find_min_norm_element(grads)

        self.model.zero_grad()
        weighted_loss = torch.mul(torch.stack(losses), sol).sum()
        weighted_loss.backward()

        return sol.detach().cpu().numpy()
