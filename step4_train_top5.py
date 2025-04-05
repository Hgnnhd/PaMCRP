import time
from config import *
from collections import Counter
from module.radam_optimizer import *
from module.net import *
from torch.utils.data import DataLoader
from module.eval import *
from module.model import *
import torch.optim as optim
from module.utils import *

seed_torch(2023)
epochs = 100
batch_size = 1024
num_classes = len(evaluation_times)
lr = 5e-4
num_layers = 1 # LSTM层数
input_dim = DIMS
hidden_dim = 256
output_dim = num_classes
max_seq_length = 241
num_encoder_layers = 3
nhead = 16
patience = 10 # 早停耐心值
counter = 10  # 早停计数器
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

train_top5_label = load_data(f'../{DATASET_DIR}/train_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_top5_label = load_data(f'../{DATASET_DIR}/validation_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_top5_label = load_data(f'../{DATASET_DIR}/test_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

# 检查重采样后的数据集大小
print("\n检查数据集大小...")
print(f"训练集大小: {len(train_features)}")
print(f"验证集大小: {len(validation_features)}")
print(f"测试集大小: {len(test_features)}")

# 检查重采样后的标签分布
print("\n检查CANCER标签分布...")
print(f"训练集标签分布: {dict(Counter(train_label))}")
print(f"验证集标签分布: {dict(Counter(validation_label))}")
print(f"测试集标签分布: {dict(Counter(test_label))}")

print("\n检查TOP5标签分布...")
print(f"训练集标签分布: {dict(sorted(Counter(train_top5_label).items()))}")
print(f"验证集标签分布: {dict(sorted(Counter(validation_top5_label).items()))}")
print(f"测试集标签分布: {dict(sorted(Counter(test_top5_label).items()))}")
print("Data loaded successfully.")


def train(model, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    for step, (inputs, aux, labels, masks, time, gold, top5_gold) in enumerate(train_loader):
        # 将数据转移到设备
        inputs = inputs.long().to(device)
        aux = aux.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        time = time.to(device)
        gold = gold.long().to(device)
        top5_gold = top5_gold.long().to(device)

        # 获取模型输出
        outputs, cancer_pred, top5_cancer_pred, proxy_loss = model(inputs, time, aux, top5_gold)
        loss_main = 0
        ranking_loss = 0
        loss_balance = []
        loss_balance.append(proxy_loss)
        sigma = 0.8
        margin = 0

        for i in range(5):  # 遍历五种癌症
            is_current_class_or_normal = (gold == (i + 1)) | (gold == 0)
            current_outputs = outputs[i][is_current_class_or_normal]  # 风险预测分数
            current_labels = labels[is_current_class_or_normal]  # 每个时间间隔是否发生癌症
            current_masks = masks[is_current_class_or_normal]  # 指示当前输出是否有效
            current_gold_subset = gold[is_current_class_or_normal]  # 筛选后的子集gold标签

            # 计算二元交叉熵损失（原始逻辑不变）
            loss = F.binary_cross_entropy_with_logits(
                current_outputs,
                current_labels.float(),
                weight=current_masks,
                reduction='none'
            )

            # 对有效样本进行加权平均（原始逻辑不变）
            valid_samples = current_masks.sum()
            if valid_samples > 0:
                average_loss = (loss.sum() / valid_samples) / 5
                loss_main += average_loss


            # 获取最后一个有效时间点的索引（子集内）
            time_weights = torch.arange(current_masks.size(1), 0, -1, device=device)  # 逆序权重
            vaild_index = torch.argmax(current_masks * time_weights, dim=1)  # (sub_batch,)

            # 提取风险值：current_outputs[sub_batch_index, time_step]
            valid_risk = current_outputs[torch.arange(len(vaild_index)), vaild_index]  # (sub_batch,)

            # 筛选癌症和正常样本（基于子集gold）
            cancer_mask_subset = (current_gold_subset == (i + 1))
            normal_mask_subset = (current_gold_subset == 0)

            cancer_valid_risk = valid_risk[cancer_mask_subset]  # 当前癌症患者的确诊风险
            normal_valid_risk = valid_risk[normal_mask_subset]  # 正常人群的最后随访风险

            #
            # # 修正对比损失计算（关键修改点）
            # if len(cancer_valid_risk) > 0 and len(normal_valid_risk) > 0:
            #     # 构造所有可能的癌症-正常风险对
            #     risk_diff = cancer_valid_risk.unsqueeze(0) - normal_valid_risk.unsqueeze(1)  # (n_cancer, n_normal)
            #     current_ranking_loss = torch.exp(-risk_diff/sigma).mean()
            #     ranking_loss += current_ranking_loss


            # 计算所有对的风险差异
            if len(cancer_valid_risk) > 0 and len(normal_valid_risk) > 0:
                risk_diff = normal_valid_risk.unsqueeze(0) - cancer_valid_risk.unsqueeze(1)  # (n_cancer, n_normal)

                # 2. 定义难样本选择比例(比如选择30%的最困难样本)
                hard_ratio = 0.1

                # 3. 计算要选择的困难样本数量
                total_pairs = risk_diff.numel()
                num_hard_samples = max(int(total_pairs * hard_ratio), 1)  # 至少选择一个样本

                # 4. 选择最困难的样本对
                with torch.no_grad():
                    # 将风险差展平为一维张量
                    flat_risk_diff = risk_diff.view(-1)

                    # Check if flat_risk_diff is empty
                    if flat_risk_diff.numel() > 0:
                        # 选择最小的num_hard_samples个风险差及其索引
                        hard_values, hard_indices = torch.topk(flat_risk_diff, k=min(num_hard_samples, total_pairs), largest=False)

                        # 5. 为不同难度的样本分配动态权重
                        # 获取所有样本的风险差范围
                        min_diff = flat_risk_diff.min()
                        max_diff = flat_risk_diff.max()
                        # 防止除以零
                        diff_range = max(max_diff - min_diff, 1e-8)

                        # 创建与flat_risk_diff相同形状的权重张量，初始化为1
                        weights = torch.ones_like(flat_risk_diff)

                        # 为困难样本分配更高权重
                        # 规范化风险差到[0,1]区间并反转，使得风险差小的样本权重大
                        norm_diffs = 1.0 - (flat_risk_diff - min_diff) / diff_range
                        # 应用非线性变换增强区分度
                        weights = torch.pow(norm_diffs, 2.0)  # 平方函数使权重分布更加突出困难样本

                        # 6. 计算加权损失
                        base_loss = torch.exp(-flat_risk_diff/sigma)
                        weighted_loss = base_loss * weights

                        hard_loss = base_loss[hard_indices] * weights[hard_indices]
                        current_ranking_loss = hard_loss.mean()

                        # 修正对比损失计算（关键修改点）
                        ranking_loss += current_ranking_loss
                    else:
                        # Handle empty case
                        current_ranking_loss = torch.tensor(0.0, device=device)
                        ranking_loss += current_ranking_loss


        # 标准化排序损失
        loss_balance.append(ranking_loss)
        # 计算分类损失
        loss_t = F.cross_entropy(top5_cancer_pred, top5_gold)
        loss_balance.append(loss_main)
        loss_balance.append(loss_t)

        optimizer.zero_grad()
        loss_final = loss_main + loss_t + proxy_loss + ranking_loss
        # 标准反向传播
        loss_final.backward()
        optimizer.step()

        running_loss += loss_final.item()

    return running_loss / len(train_loader)


train_dataset = CustomDataset_aux(train_features, train_aux, train_labels, train_masks, train_times, train_label,train_top5_label)
validation_dataset = CustomDataset_aux(validation_features, validation_aux, validation_labels, validation_masks, validation_times, validation_label,validation_top5_label)
test_dataset = CustomDataset_aux(test_features, test_aux, test_labels, test_masks, test_times, test_label, test_top5_label)
# 使用DataLoader加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
# 定义模型
model = Mymodel(input_dim, hidden_dim, output_dim, nhead, max_seq_length, num_encoder_layers, num_classes, model_type).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
params = list(model.parameters())
Patero = MGDA(model, task_num=3, device=device)
if model_type == "hnn":
    optimizer = RiemannianAdam(params, lr=lr)
else:
    optimizer = optim.Adam(params, lr=lr)

#训练模型
print(f"Training {model.__class__.__name__} with {model_type} and {num_classes} classes")
pbar = tqdm(range(epochs))
best_val_auc = 0.0
best_epoch = 0
for epoch in pbar:
    a = time.time()
    loss = train(model, train_dataloader, optimizer)
    results, roc_data = evaluate_top5(model, test_dataloader)
    pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
    train_time = time.time() - a
    # Calculate average AUC and CI for each category
    avg_auc_categories = {}
    avg_auc_ci_categories = {}
    for category in range(1, 6):
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
    for cat in range(1, 6):
        auc = avg_auc_categories[f'Avg_AUC_{cat}']
        ci_lower, ci_upper = avg_auc_ci_categories[f'Avg_AUC_CI_{cat}']
        postfix[f'AUC_{cat}'] = f"{auc:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
    # Add C-index and CI to postfix
    for category in range(1, 6):
        c_index = results.get(f"C_index_{category}", "N/A")
        c_index_ci = results.get(f"C_index_CI_{category}", ("N/A", "N/A"))
        if isinstance(c_index, float) and isinstance(c_index_ci[0], float) and isinstance(c_index_ci[1], float):
            postfix[f"C-index_{category}"] = f"{c_index:.4f} ({c_index_ci[0]:.4f}-{c_index_ci[1]:.4f})"
    pbar.set_postfix(postfix)
    # Save the best performing model
    if avg_auc_roc > best_val_auc:
        best_val_auc = avg_auc_roc
        best_model_state_dict = model.state_dict()
        #加入时间戳
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f'../results/best_model_{model_type}_{current_time}.pth'
        torch.save(best_model_state_dict, name)
        best_epoch = epoch + 1
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        break
# 评估模型
model.load_state_dict(torch.load(name))  # 加载性能最好的模型参数
results, roc_data = evaluate_top5(model, test_dataloader, test=True)

others = f"{EXCLUDE_DAY}"
# 打印所有时间点的主要指标
print_results(results)
results_filename = save_training_results(results, name, './result/my_results', others)
print(f"Training results saved as {results_filename}")
print("best model saved as", name)



