import torch.nn
import torch.optim as optim
from time import time
from model import *
from config import *
from collections import Counter
from tqdm import tqdm
import os
from radam_optimizer import *
from torch.utils.data import DataLoader
from eval import *
# 获取当前脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 将工作目录更改为脚本所在目录
os.chdir(script_dir)
seed_torch(2023)

epochs = 30
batch_size = 1024
num_classes = len(evaluation_times)
lr = 5e-4
num_layers = 1 # LSTM层数
input_dim = DIMS
hidden_dim = 256
output_dim = num_classes
max_seq_length = 241
num_encoder_layers = 2
nhead = 16
patience = 10
counter = 0

print(f"正在进行{TARGET_PATTERN}的{len(evaluation_times)}年内风险预测")
print("加载数据中")
# 加载训练集、验证集和测试集的特征、邻接矩阵和标签
train_features = load_data(f'../{DATASET_DIR}/train_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_features = load_data(f'../{DATASET_DIR}/validation_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_features = load_data(f'../{DATASET_DIR}/test_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_aux = load_data(f'../{DATASET_DIR}/train_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_aux = load_data(f'../{DATASET_DIR}/validation_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_aux = load_data(f'../{DATASET_DIR}/test_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_times = load_data(f'../{DATASET_DIR}/train_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_times = load_data(f'../{DATASET_DIR}/validation_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_times = load_data(f'../{DATASET_DIR}/test_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_labels = load_data(f'../{DATASET_DIR}/train_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_labels = load_data(f'../{DATASET_DIR}/validation_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_labels = load_data(f'../{DATASET_DIR}/test_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_label = load_data(f'../{DATASET_DIR}/train_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_label = load_data(f'../{DATASET_DIR}/validation_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_label = load_data(f'../{DATASET_DIR}/test_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_masks = load_data(f'../{DATASET_DIR}/train_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_masks = load_data(f'../{DATASET_DIR}/validation_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_masks = load_data(f'../{DATASET_DIR}/test_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

print("数据加载成功")

##数据集平衡
print("\n正在进行数据集平衡...")
t_dim = len(train_features[0])
# 检查重采样后的数据集大小
print("\n检查重采样后的数据集大小...")
print(f"重采样后训练集大小: {len(train_features)}")
print(f"重采样后验证集大小: {len(validation_features)}")
print(f"重采样后测试集大小: {len(test_features)}")

# 检查重采样后的标签分布
print("\n检查重采样后的标签分布...")
print(f"重采样后训练集标签分布: {dict(Counter(train_label))}")
print(f"重采样后验证集标签分布: {dict(Counter(validation_label))}")
print(f"重采样后测试集标签分布: {dict(Counter(test_label))}")


class Mymodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, t_dim, nhead, max_seq_length, num_encoder_layers, num_classes):
        super(Mymodel, self).__init__()

        self.risk_factor = False
        self.age_factor = False
        self.sex_factor = False

        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t_dim = t_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes


        self.lstm = LSTM(input_dim, hidden_dim, num_layers=1,output_size=input_dim)
        self.lstm_ = LSTM(hidden_dim, hidden_dim, num_layers=1, output_size=input_dim)

        self.bilstm = BiLSTMModel(input_dim, hidden_dim, num_layers=1)
        self.bilstm_ = BiLSTMModel(hidden_dim, hidden_dim, num_layers=1)

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.fc_t = nn.Linear(t_dim, 1)

        self.trans = Transformer(input_dim, hidden_dim, hidden_dim, nhead, max_seq_length, num_encoder_layers)
        self.trans_ = Transformer(hidden_dim, hidden_dim, hidden_dim, nhead, max_seq_length, num_encoder_layers)

        self.tcn = TCN(input_dim,hidden_dim, num_channels=[32,64], kernel_size=15, dropout=0.5)

        self.hnn = HNN(input_dim, hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.batch_norm = nn.BatchNorm1d(max_seq_length)
        self.fc_sex = nn.Linear(1,hidden_dim)
        self.fc_3 = nn.Linear(1,num_classes)

        self.projector = nn.Sequential(nn.Linear(hidden_dim, 2))
        self.classifier_top5 = nn.Sequential(nn.Linear(hidden_dim, 6))

        self.fft = freq_mix

        self.sex_emb = nn.Embedding(2, hidden_dim+1) if self.age_factor else nn.Embedding(2, hidden_dim)

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
        if self.risk_factor:
            self.prob_of_failure_layer = CumulativeProbabilityLayer(hidden_dim + 5, len(evaluation_times))
            self.prob_of_failure_layer_male = CumulativeProbabilityLayer(hidden_dim + 5, len(evaluation_times))
            self.prob_of_failure_layer_female = CumulativeProbabilityLayer(hidden_dim + 5, len(evaluation_times))
            self.prob_of_failure_layer_1 = CumulativeProbabilityLayer(hidden_dim+5, len(evaluation_times))
            self.prob_of_failure_layer_2 = CumulativeProbabilityLayer(hidden_dim+5, len(evaluation_times))
            self.prob_of_failure_layer_3 = CumulativeProbabilityLayer(hidden_dim+5, len(evaluation_times))
            self.prob_of_failure_layer_4 = CumulativeProbabilityLayer(hidden_dim+5, len(evaluation_times))
            self.prob_of_failure_layer_5 = CumulativeProbabilityLayer(hidden_dim+5, len(evaluation_times))
        elif self.age_factor:
            self.prob_of_failure_layer = CumulativeProbabilityLayer(hidden_dim + 1, len(evaluation_times))
            self.prob_of_failure_layer_male = CumulativeProbabilityLayer(hidden_dim + 1, len(evaluation_times))
            self.prob_of_failure_layer_female = CumulativeProbabilityLayer(hidden_dim + 1, len(evaluation_times))
            self.prob_of_failure_layer_1 = CumulativeProbabilityLayer(hidden_dim+1, len(evaluation_times))
            self.prob_of_failure_layer_2 = CumulativeProbabilityLayer(hidden_dim+1, len(evaluation_times))
            self.prob_of_failure_layer_3 = CumulativeProbabilityLayer(hidden_dim+1, len(evaluation_times))
            self.prob_of_failure_layer_4 = CumulativeProbabilityLayer(hidden_dim+1, len(evaluation_times))
            self.prob_of_failure_layer_5 = CumulativeProbabilityLayer(hidden_dim+1, len(evaluation_times))
        else:
            self.prob_of_failure_layer = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
            self.prob_of_failure_layer_male = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
            self.prob_of_failure_layer_female = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
            self.prob_of_failure_layer_1 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
            self.prob_of_failure_layer_2 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
            self.prob_of_failure_layer_3 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
            self.prob_of_failure_layer_4 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))
            self.prob_of_failure_layer_5 = CumulativeProbabilityLayer(hidden_dim, len(evaluation_times))


        MAX_TIME_EMBED_PERIOD_IN_DAYS = 120 * 365
        MIN_TIME_EMBED_PERIOD_IN_DAYS = 1

        self.multipliers = 2 * math.pi / torch.linspace(
            start=MIN_TIME_EMBED_PERIOD_IN_DAYS,
            end=MAX_TIME_EMBED_PERIOD_IN_DAYS,
            steps=self.time_embed_dim
        ).view(1, 1, -1)


        self.non_linear = nn.Sequential(
            nn.Linear(input_dim,input_dim),
            nn.ReLU(),
            nn.Linear(input_dim,input_dim)
        )

        num_prompts = 2
        # 为每个prompt分配可学习的参数
        proxy_dim = 6
        self.proxy = nn.Parameter(torch.randn(num_prompts, proxy_dim))
        self.GAT = GAT(input_dim, hidden_dim, hidden_dim, dropout=0.2, alpha=0.2, nheads=4)
        self.fc_proxy = nn.Linear(hidden_dim+proxy_dim, hidden_dim)
        self.projector_proxy = nn.Sequential(nn.Linear(hidden_dim, proxy_dim))
        self.proxy_k = nn.Parameter(torch.randn(num_prompts, hidden_dim))
        self.gram_schmidt_single_vector(self.proxy)

    def gram_schmidt_single_vector(self, vv):
        def projection(u, v):
            denominator = (u * u).sum()
            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u
        if len(vv.shape) == 3:
            vv = torch.mean(vv, dim=1)
        # 确保输入是2D张量，形状为 [N, dim]，其中N是任务数量
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
                            print('重新开始!!!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[k] = vk - uk

        # 单位化每个向量
        for k in range(N):
            uu[k] = uu[k] / uu[k].norm()

        return torch.nn.Parameter(uu)

    def age_prompt(self, age):
        age_float = age.float().unsqueeze(-1) / 120.0  # 假设最大年龄是120岁

        # 使用 torch.arange 和 torch.pow 来避免整数溢出
        i = torch.arange(self.dim // 2, dtype=torch.float32, device=age.device)
        freqs = torch.pow(2.0, i)

        # 计算 sin 和 cos 项
        sin_terms = torch.sin(age_float * freqs.unsqueeze(0))
        cos_terms = torch.cos(age_float * freqs.unsqueeze(0))

        # 连接 sin 和 cos 项
        age_enc = torch.cat([sin_terms, cos_terms], dim=-1)

        return age_enc.squeeze()

    def InfoNCE_loss(self, x_embed, prompt_embed, positive_indices):
        if x_embed.dim() == 3:
            x_embed = torch.mean(x_embed, dim=1)

        x_embed = self.projector_proxy(x_embed)

        ortholoss = OrthogonalityLosses.flexible_orthogonal_loss(prompt_embed)

        x_embed = F.normalize(x_embed, dim=1)
        prompt_embed = F.normalize(prompt_embed, dim=1)

        # 计算 x 和所有 prompt 之间的相似度
        similarity = torch.matmul(x_embed, prompt_embed.T)

        # 提取正例的相似度
        positive_sim = similarity[torch.arange(similarity.size(0)), positive_indices]

        # 将正例从相似度矩阵中移除
        similarity_without_positive = similarity.clone()
        similarity_without_positive[torch.arange(similarity.size(0)), positive_indices] = float('-inf')

        # 计算 exclusive InfoNCE loss
        loss = -positive_sim + torch.logsumexp(similarity_without_positive, dim=1)
        return loss.mean() + ortholoss

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
            # 计算注意力图
            similarity = torch.matmul(x_norm, prompt_norm.t())
            #计算权重softmax
            _, positive_indices = similarity.max(dim=-1)  # shape: (batch_size * seq_len,)
            proxy = proxy_emb[positive_indices]
        else:
            proxy_main = proxy_emb[positive_indices]
            proxy = proxy_main
        # 计算 InfoNCE loss
        if positive_indices is None:
            proxy_loss = 0.0
        else:
            proxy_loss = self.InfoNCE_loss(x, proxy_emb, positive_indices)
        # 连接x和prompt_embed
        x_prompt = torch.cat([x, proxy], dim=-1)
        x_prompt = self.fc_proxy(x_prompt)

        return x_prompt, proxy_loss

    def condition_on_pos_embed(self, x, embed, embed_type='time'):
        if embed_type == 'time':
            return self.t_embed_scale_fc(embed) * x + self.t_embed_add_fc(embed)
        elif embed_type == 'age':
            return self.a_embed_scale_fc(embed) * x + self.a_embed_add_fc(embed)
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

    def embedding_layer(self, x, single=False):

        batch, seq, dim = x.size()

        x_1 = self.emb_1(x[:, 0, :])
        x_2 = self.emb_2(x[:, 1, :])
        x_3 = self.emb_3(x[:, 2, :])
        x_4 = self.emb_4(x[:, 3, :])
        if single:
            return x_4
        # Combine embeddings
        x_combined = torch.stack((x_1, x_2, x_3, x_4), dim=2)  # (batch, 4, hidden_dim)
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

    def forward(self, x, time, aux, prompt=None):

        x_copy = x
        batch, seq, _ = x.size()
        x = self.embedding_layer(x,single=True)

        t_different = (time[:, 0, :])  # 每两个疾病间隔时间
        t_last = (time[:, 1, :])  # 跟最后一次间隔时间
        t_age = (time[:, 2, :])  # 每次疾病的年龄
        age = (torch.max(t_age,dim=1)[0] / 365.0).unsqueeze(-1) ##当前年龄

        sex = aux[:, 0].long()
        time_emb_1 = self.get_time_seq(t_different)
        time_emb_2 = self.get_time_seq(t_last)
        age_emb = self.get_time_seq(t_age)

        x = self.condition_on_pos_embed(x, time_emb_1, embed_type='time')
        x = self.condition_on_pos_embed(x, time_emb_2, embed_type='time')
        x = self.condition_on_pos_embed(x, age_emb, embed_type='age')
        x = self.pos(x)

        if model_type == "mlp":
            x = self.fc(x)
            x = torch.mean(x, dim=1)

        elif model_type == "tcn":
            x = self.tcn(x)
            x = torch.mean(x, dim=1)

        elif model_type == "lstm":
            x = self.bilstm(x)
            x = torch.mean(x, dim=1)

        elif model_type == "transformer":
            x = self.trans(x)
            x = torch.mean(x, dim=1)

        elif model_type == "hnn":
            x, z = self.hnn(x_copy)

        x_cancer = self.projector(x)
        x_top5_cancer = self.classifier_top5(x)
        proxy_loss = 0
        # 使用可学习的prompt嵌入
        # x, proxy_loss = self.proxy_loss(x, prompt)

        if self.risk_factor:
            x = torch.cat((x, aux), -1)
        elif self.age_factor:
            x = torch.cat((x, age), -1)

        x = self.prob_of_failure_layer(x)

        return x, x_cancer, x_top5_cancer, proxy_loss

def train(model, train_loader, optimizer, criterion):

    model.train()
    running_loss = 0.0

    for inputs, aux, labels, masks, time, gold in train_loader:
        inputs = inputs.long()
        optimizer.zero_grad()
        outputs, cancer_pred, z2, proxy_loss = model(inputs, time, aux, gold)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float(), weight=masks, reduction='sum')
        loss = loss / torch.sum(masks.float())

        loss_c = F.cross_entropy(cancer_pred, gold)

        loss_final = (loss + loss_c +  proxy_loss) / 3

        loss_final.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def print_results(results, metrics_to_display=None):
    output = []
    time_points = list(set([key.split('_')[1] for key in results.keys() if key.startswith('Time_')]))
    time_points = sorted(time_points, key=int)

    category_names = {
        0: "癌",
    }

    # Print AUC results
    output.append(f"\nCategory-specific AUC Results for {TARGET_PATTERN}:")
    categories = range(0, 1)
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


def save_training_results(results, filename_prefix):
    # 为了加入日期时间戳
    num_classes = len(evaluation_times) - 2
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 构建保存文件名
    results_filename = f'{filename_prefix}_{current_time}_{num_classes}_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{model_type}.json'

    # 使用 JSON 文件保存训练结果
    with open(results_filename, 'w') as json_file:
        json.dump(results, json_file)

    # 生成打印结果的字符串
    printed_results = print_results(results)

    # 在控制台显示结果
    print(printed_results)

    # 保存打印结果到文本文件
    txt_filename = f'{filename_prefix}_{current_time}_{num_classes}_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{model_type}_printed_results.txt'
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(printed_results)

    return results_filename, txt_filename

def collate_fn(batch):
    # 解包批次数据
    data, aux, label, masks, time, gold, *_ = zip(*batch)
    # 找出最大长度
    max_len = max(d.shape[1] for d in data)
    # 填充数据到最大长度
    padded_data = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in data]
    padded_time = [torch.nn.functional.pad(d, (0, max_len - d.shape[1])) for d in time]

    # 堆叠所有张量
    data = torch.stack(padded_data)
    time = torch.stack(padded_time)
    aux = torch.stack(aux)
    label = torch.stack(label)
    masks = torch.stack(masks)
    gold = torch.stack(gold)

    return data, aux, label, masks, time, gold


train_dataset = CustomDataset_aux(train_features, train_aux, train_labels, train_masks, train_times, train_label)
validation_dataset = CustomDataset_aux(validation_features, validation_aux, validation_labels,validation_masks, validation_times, validation_label)
test_dataset = CustomDataset_aux(test_features, test_aux, test_labels, test_masks, test_times, test_label)

# 使用DataLoader加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 创建数据加载器
model = Mymodel(input_dim, hidden_dim, output_dim, t_dim, nhead, max_seq_length, num_encoder_layers, num_classes).to(device)
metaloss = MetaLoss()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(metaloss.parameters())

if model_type == "hnn":
    optimizer = RiemannianAdam(params, lr=lr)
else:
    optimizer = optim.Adam(params, lr=lr)

# 训练模型
##说明现在用了什么模型和参数
print(f"Training {model.__class__.__name__} with {model_type} and {num_classes} classes")
pbar = tqdm(range(epochs))
best_val_auc = 0.0
best_epoch = 0
loss = 0
print("开始训练")
for epoch in pbar:
    a = time()
    loss = train(model, train_dataloader, optimizer, metaloss)
    results = evaluate_single_category(model, validation_dataloader)

    pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
    train_time = time() - a

    # Calculate average AUC and CI for each category
    avg_auc_categories = {}
    avg_auc_ci_categories = {}
    for category in range(0, 1):
        avg_auc = sum(results.get(f'Time_{t}_{category}_AUC', 0) for t in evaluation_times) / len(
            evaluation_times)
        avg_auc_categories[f'Avg_AUC_{category}'] = avg_auc

        # Calculate average CI
        ci_lowers = [results.get(f'Time_{t}_{category}_AUC_CI', (0, 0))[0] for t in evaluation_times]
        ci_uppers = [results.get(f'Time_{t}_{category}_AUC_CI', (0, 0))[1] for t in evaluation_times]
        avg_ci_lower = sum(ci_lowers) / len(evaluation_times)
        avg_ci_upper = sum(ci_uppers) / len(evaluation_times)
        avg_auc_ci_categories[f'Avg_AUC_CI_{category}'] = (avg_ci_lower, avg_ci_upper)

    # Set tqdm progress bar suffix
    postfix = {
        "best_epoch": f"{best_epoch:.1f}",
        "best_auc": f"{best_val_auc:.4f}",
        'Loss': f"{loss:.4f}",
        'Time': f"{train_time:.2f}s",
    }

    avg_auc_roc = sum(avg_auc_categories.values()) / len(avg_auc_categories)
    postfix['Avg AUC'] = f"{avg_auc_roc:.4f}"

    for cat in range(0, 1):
        auc = avg_auc_categories[f'Avg_AUC_{cat}']
        ci_lower, ci_upper = avg_auc_ci_categories[f'Avg_AUC_CI_{cat}']
        postfix[f'AUC_{cat}'] = f"{auc:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"

    # Add C-index and CI to postfix
    for category in range(0, 1):
        c_index = results.get(f"C_index_{category}", "N/A")
        c_index_ci = results.get(f"C_index_CI_{category}", ("N/A", "N/A"))
        if isinstance(c_index, float) and isinstance(c_index_ci[0], float) and isinstance(c_index_ci[1], float):
            postfix[f"C-index_{category}"] = f"{c_index:.4f} ({c_index_ci[0]:.4f}-{c_index_ci[1]:.4f})"

    pbar.set_postfix(postfix)

    # Save the best performing model
    if avg_auc_roc > best_val_auc:
        best_val_auc = avg_auc_roc
        best_model_state_dict = model.state_dict()
        # 加入时间戳
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f'../results/best_model_{model_type}_{current_time}.pth'
        torch.save(best_model_state_dict, name)
        best_epoch = epoch + 1
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        break

print("训练完成")
print(f"最好的验证集AUC-ROC: {best_val_auc:.4f}")
# 评估模型
model.load_state_dict(torch.load(name))  # 加载性能最好的模型参数
# 评估模型

results = evaluate_single_category(model, test_dataloader, test=True)

# 打印所有时间点的主要指标
print_results(results)

# 保存训练结果
results["censor_date"] = EXCLUDE_DAY
results["best_val_auc"] = best_val_auc

results_filename = save_training_results(results, './result/my_results')
print(f"Training results saved as {results_filename}")


"""

C50
Metric                       Time_365    Time_730   Time_1095   Time_1460   Time_1825
-------------------------------------------------------------------------------------
AUC-ROC                        0.7875      0.7512      0.7285      0.6974      0.6995
Metric                       Time_365    Time_730   Time_1095   Time_1460   Time_1825

average = (78.75 + 75.12 + 72.85 + 69.74 + 69.95) / 5 = 73.28

C61
Metric                       Time_365    Time_730   Time_1095   Time_1460   Time_1825
-------------------------------------------------------------------------------------
AUC-ROC                        0.8286      0.7992      0.7757      0.7671      0.7721

average = (82.86 + 79.92 + 77.57 + 76.71 + 77.21) / 5 = 78.85

C34
Metric                       Time_365    Time_730   Time_1095   Time_1460   Time_1825
-------------------------------------------------------------------------------------
AUC-ROC                        0.9389      0.9227      0.9147      0.9271      0.9229

average = (93.89 + 92.27 + 91.47 + 92.71 + 92.29) / 5 = 92.52

C16
Metric                       Time_365    Time_730   Time_1095   Time_1460   Time_1825
-------------------------------------------------------------------------------------
AUC-ROC                        0.9379      0.9351      0.9083      0.9067      0.9131

average = (93.79 + 93.51 + 90.83 + 90.67 + 91.31) / 5 = 92.02

最好的验证集AUC-ROC: 0.9225

C18-C20
Metric                       Time_365    Time_730   Time_1095   Time_1460   Time_1825
-------------------------------------------------------------------------------------
AUC-ROC                        0.9526      0.9486      0.9456      0.9427      0.9238

average = (95.26 + 94.86 + 94.56 + 94.27 + 92.38) / 5 = 94.26

"""