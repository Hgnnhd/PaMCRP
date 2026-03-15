import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module.model import Transformer, CumulativeProbabilityLayer
from module.utils import OrthogonalityLosses
from config import TARGET_PATTERN, evaluation_times, device


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

        x_top5_cancer = self.classifier_top5(x_hidden)

        x_1 = self.prob_of_failure_layer_1(x)
        x_2 = self.prob_of_failure_layer_2(x)
        x_3 = self.prob_of_failure_layer_3(x)
        x_4 = self.prob_of_failure_layer_4(x)
        x_5 = self.prob_of_failure_layer_5(x)

        x = torch.stack([x_1,x_2,x_3,x_4,x_5],dim=0)

        return x, x_top5_cancer, proxy_loss