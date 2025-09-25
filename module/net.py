from .utils import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            dmin = 1e8
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

            skippers = torch.sum(tm1 < 1e-7) + torch.sum(tm2 < 1e-7)
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
            # This is optimal for n=2, so return the solution
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
        grads = grads / gn.unsqueeze(1).repeat(1, grads.size()[1])
        return grads

    def backward(self, losses):
        grads = self._compute_grad(losses)
        loss_data = torch.tensor([loss.item() for loss in losses]).to(self.device)
        grads = self._gradient_normalizers(grads, loss_data, ntype="l2")
        sol = self._find_min_norm_element(grads)

        self.model.zero_grad()
        weighted_loss = torch.mul(torch.stack(losses), sol).sum()
        weighted_loss.backward()

        return sol.detach().cpu().numpy(), weighted_loss.detach()


class Patero(MGDA):
    """
    Patero optimizer wrapper using MGDA (minimum-norm) implementation
    for Pareto-style multi-objective gradient balancing.

    This class exists to reflect the project naming (Patero) while
    reusing the MGDA algorithm as the selected implementation.
    """
    pass


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

    def hazards(self, x, prompt=None,age=None):
        # x shape: (batch_size, num_features)
        # prompt shape: (1, num_features)
        raw_hazard = self.hazard_fc(x)  # (batch_size, max_followup)
        if prompt is not None:
            prompt_proj = self.prompt_fc(prompt)  # (1, max_followup)
            prompt_proj = prompt_proj.expand(raw_hazard.size(0), -1)  # (batch_size, max_followup)
            combined = torch.cat([raw_hazard, prompt_proj], dim=-1)  # (batch_size, max_followup * 2)
            raw_hazard = self.fusion_fc(combined)  # (batch_size, max_followup)

        pos_hazard = self.relu(raw_hazard)

        return pos_hazard

    def forward(self, x, prompt=None):
        hazards = self.hazards(x, prompt)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # expanded_hazards is (B,T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask  # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)

        return cum_prob


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

    def forward(self, x,key_padding_mask=None):
        h = self.multihead_attention(x,key_padding_mask)
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

    def forward(self, x, key_padding_mask=None):
        for layer in self.encoder_layers:
            x = layer(x,key_padding_mask)
        x = self.fc_out(x)
        return x

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

    def attention(self, q, k, v, key_padding_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_per_head)

        if key_padding_mask is not None:
            # key_padding_mask shape: (B, N_k)
            # Expand mask to broadcast: (B, 1, 1, N_k)
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # Fill with negative infinity where mask is True
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, x, key_padding_mask=None):
        B, N, H = x.size()

        # perform linear operation and split into h heads
        k = self.key(x).view(B, N, self.num_heads, self.dim_per_head)
        q = self.query(x).view(B, N, self.num_heads, self.dim_per_head)
        v = self.value(x).view(B, N, self.num_heads, self.dim_per_head)

        # transpose to get dimensions B * args.num_heads * S * dim_per_head
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        h = self.attention(q, k, v, key_padding_mask)

        # concatenate heads and put through final linear layer
        h = h.transpose(1, 2).contiguous().view(B, -1, H)

        output = self.aggregate_fc(h)

        return output

class PaMCRP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, max_seq_length, num_encoder_layers, num_classes, model_type):
        super(PaMCRP, self).__init__()
        self.risk_factor = False
        self.age_factor = False
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.trans = Transformer(input_dim, hidden_dim, hidden_dim, nhead, max_seq_length, num_encoder_layers)

        self.relu = nn.ReLU()

        self.projector = nn.Sequential(nn.Linear(hidden_dim, 2))
        self.classifier_top5 = nn.Sequential(nn.Linear(hidden_dim, 6))

        self.emb = nn.Embedding(9999, input_dim)
        self.emb_1 = nn.Embedding(9999, input_dim)
        self.emb_2 = nn.Embedding(9999, input_dim)
        self.emb_3 = nn.Embedding(9999, input_dim)
        self.emb_4 = nn.Embedding(9999, input_dim)

        self.time_embed_dim = input_dim
        self.t_embed_add_fc = nn.Linear(input_dim, input_dim)
        self.t_embed_scale_fc = nn.Linear(input_dim, input_dim)

        self.a_embed_add_fc = nn.Linear(input_dim, input_dim)
        self.a_embed_scale_fc = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(0.2)

        MAX_TIME_EMBED_PERIOD_IN_DAYS = 120 * 365
        MIN_TIME_EMBED_PERIOD_IN_DAYS = 1

        self.multipliers = 2 * math.pi / torch.linspace(
            start=MIN_TIME_EMBED_PERIOD_IN_DAYS,
            end=MAX_TIME_EMBED_PERIOD_IN_DAYS,
            steps=self.time_embed_dim
        ).view(1, 1, -1)

        self.non_linear = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

        num_prompts = len(TARGET_PATTERN) - 1

        proxy_dim = num_prompts
        self.proxy = nn.Parameter(torch.randn(num_prompts, proxy_dim))
        self.fc_proxy = nn.Linear(hidden_dim + proxy_dim, hidden_dim)
        self.projector_proxy = nn.Sequential(nn.Linear(hidden_dim, proxy_dim))
        self.gram_schmidt_single_vector(self.proxy)

        self.demo_embed_scale_fc = nn.Linear(input_dim, input_dim)
        self.demo_embed_add_fc = nn.Linear(input_dim, input_dim)


        self.prob_of_failure_layer = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
        self.prob_of_failure_layer_male = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
        self.prob_of_failure_layer_female = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
        self.prob_of_failure_layer_1 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
        self.prob_of_failure_layer_2 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
        self.prob_of_failure_layer_3 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
        self.prob_of_failure_layer_4 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
        self.prob_of_failure_layer_5 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))

    def gram_schmidt_single_vector(self, vv):
        def projection(u, v):
            denominator = (u * u).sum()
            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u
        if len(vv.shape) == 3:
            vv = torch.mean(vv, dim=1)

        assert len(vv.shape) == 2, "Input should be a 2D tensor"
        N, dim = vv.shape

        uu = torch.zeros_like(vv, device=vv.device)

        for k in range(N):
            redo = True
            while redo:
                redo = False
                vk = torch.randn(dim, device=vv.device)
                uk = torch.zeros(dim, device=vv.device)
                for j in range(k):
                    if not redo:
                        uj = uu[j]
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restart!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[k] = vk - uk


        for k in range(N):
            uu[k] = uu[k] / uu[k].norm()

        return torch.nn.Parameter(uu)

    def InfoNCE_loss(self, x_embed, prompt_embed, positive_indices):
        if x_embed.dim() == 3:
            x_embed = torch.mean(x_embed, dim=1)

        x_embed = self.projector_proxy(x_embed)

        ortholoss = OrthogonalityLosses.flexible_orthogonal_loss(prompt_embed)

        x_embed = F.normalize(x_embed, dim=1)
        prompt_embed = F.normalize(prompt_embed, dim=1)


        similarity = torch.matmul(x_embed, prompt_embed.T)


        positive_sim = similarity[torch.arange(similarity.size(0)), positive_indices]


        similarity_without_positive = similarity.clone()
        similarity_without_positive[torch.arange(similarity.size(0)), positive_indices] = float('-inf')


        loss = -positive_sim + torch.logsumexp(similarity_without_positive, dim=1)

        return (loss.mean() + ortholoss)

    def proxy_loss(self, x, positive_indices=None):
        proxy_emb = self.proxy
        if positive_indices is None:
            if x.dim() == 3:
                x_ = torch.mean(x, dim=1)
            else:
                x_ = x
            x_ = self.projector_proxy(x_)
            x_norm = F.normalize(x_, p=2, dim=-1)
            prompt_norm = F.normalize(proxy_emb, p=2, dim=-1)
            similarity = torch.matmul(x_norm, prompt_norm.t())
            _, positive_indices = similarity.max(dim=-1)  # shape: (batch_size * seq_len,)
            proxy = proxy_emb[positive_indices]
        else:
            proxy = proxy_emb[positive_indices]


        if positive_indices is None:
            proxy_loss = 0.0
        else:
            proxy_loss = self.InfoNCE_loss(x, proxy_emb, positive_indices)

        x_prompt = torch.cat([x, proxy], dim=-1)
        x_prompt = self.fc_proxy(x_prompt)

        return x_prompt, proxy_loss

    def condition_on_pos_embed(self, x, embed, embed_type='time'):
        if embed_type == 'time':
            return self.t_embed_scale_fc(embed) * x + self.t_embed_add_fc(embed)
        elif embed_type == 'age':
            return self.a_embed_scale_fc(embed) * x + self.a_embed_add_fc(embed)
        elif embed_type == 'pos':
            return self.demo_embed_scale_fc(embed) * x + self.demo_embed_add_fc(embed)
        else:
            raise NotImplementedError("Embed type {} not supported".format(embed_type))

    def get_time_seq(self, deltas):
        """
        Calculates the positional embeddings depending on the time diff from the events and the reference date.
        """
        deltas = deltas.unsqueeze(-1)
        self.multipliers = self.multipliers.to(deltas.device)
        positional_embeddings = torch.cos(deltas * self.multipliers)
        return positional_embeddings

    def pos(self, x):
        h_dim = x.shape[-1]
        pe = torch.zeros(self.max_seq_length, h_dim)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, h_dim, 2).float() * (-math.log(10000.0) / h_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        return x + pe[:,:x.size(1), :]


    def pos_emb(self, x):
        h_dim = x.shape[-1]
        pe = torch.zeros(self.max_seq_length, h_dim)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, h_dim, 2).float() * (-math.log(10000.0) / h_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        return pe[:,:x.size(1), :]


    def embedding_layer(self, x, level=4):

        x_1 = self.emb_1(x[:, 0, :])
        x_2 = self.emb_2(x[:, 1, :])
        x_3 = self.emb_3(x[:, 2, :])
        x_4 = self.emb_4(x[:, 3, :])

        if level == 1:
            return x_4
        elif level == 2:
            x_combined = torch.stack((x_3, x_4), dim=2)  # (batch, 4, hidden_dim)
        elif level == 3:
            x_combined = torch.stack((x_2, x_3, x_4), dim=2)  # (batch, 4, hidden_dim)
        elif level == 4:
            x_combined = torch.stack((x_1, x_2, x_3, x_4), dim=2)

        # Apply attention
        output = self.attention(x_combined)# (batch, 4, hidden_dim)
        # Apply non-linear transformation
        batch, seq, level, dim = output.size()
        output = self.non_linear(output) # (batch, 4, hidden_dim)
        # Average over the channel dimension
        output = torch.mean(output, dim=2)  # (batch, hidden_dim)

        return output

    def attention(self, x):
        # Simple dot-product attention
        attn_weights = F.softmax(torch.matmul(x, x.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)
        return torch.matmul(attn_weights, x)

    def forward(self, x,  time, prompt=None):

        batch, seq, _ = x.size()
        proxy_loss = 0

        x = self.embedding_layer(x, level=4)

        t_different = (time[:, 0, :])
        t_last = (time[:, 1, :])
        t_age = (time[:, 2, :])

        time_emb_1 = self.get_time_seq(t_different)
        time_emb_2 = self.get_time_seq(t_last)
        age_emb = self.get_time_seq(t_age)


        x = self.condition_on_pos_embed(x, time_emb_1, embed_type='time')
        x = self.condition_on_pos_embed(x, time_emb_2, embed_type='time')
        x = self.condition_on_pos_embed(x, age_emb, embed_type='age')
        x = self.pos(x)
        x = self.trans(x)
        x = torch.mean(x, dim=1)
        x_hidden = x
        x, proxy_loss = self.proxy_loss(x, prompt)

        x_cancer = x

        x_top5_cancer = self.classifier_top5(x_hidden)

        x_1 = self.prob_of_failure_layer_1(x)
        x_2 = self.prob_of_failure_layer_2(x)
        x_3 = self.prob_of_failure_layer_3(x)
        x_4 = self.prob_of_failure_layer_4(x)
        x_5 = self.prob_of_failure_layer_5(x)

        x = torch.stack([x_1,x_2,x_3,x_4,x_5],dim=0)

        return x, x_cancer, x_top5_cancer, proxy_loss

