import pickle
from collections import Counter
from config import *

# 加载数据集
dataset = pickle.load(open(f"../{DATASET_DIR}/patient_dataset_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}_aux.pkl", "rb"))
features, aux, time, label = zip(*dataset)

labels = pickle.load(open(f"../{DATASET_DIR}/label_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb"))
masks = pickle.load(open(f"../{DATASET_DIR}/mask_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb"))
top5_label = pickle.load(open(f"../{DATASET_DIR}/top5_label_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb"))

# 读取idx
with open(f"../{DATASET_DIR}/train_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    train_idx = pickle.load(f)

with open(f"../{DATASET_DIR}/val_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    val_idx = pickle.load(f)

with open(f"../{DATASET_DIR}/test_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    test_idx = pickle.load(f)



# 转换为集合以便于操作
train_set = set(train_idx)
val_set = set(val_idx)
test_set = set(test_idx)

# 检查索引的有效性
all_indices = set(range(len(features)))
used_indices = train_set | val_set | test_set

print(f"训练集大小: {len(train_set)}")
print(f"验证集大小: {len(val_set)}")
print(f"测试集大小: {len(test_set)}")
print(f"总使用索引数: {len(used_indices)}")
print(f"特征总数: {len(features)}")

# 进行各种检查...

# 构建数据集
train_features = [features[i] for i in train_idx]
train_aux = [aux[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
train_label = [label[i] for i in train_idx]
train_masks = [masks[i] for i in train_idx]
train_times = [time[i] for i in train_idx]
train_top5_label = [top5_label[i] for i in train_idx]


validation_features = [features[i] for i in val_idx]
validation_aux = [aux[i] for i in val_idx]
validation_labels = [labels[i] for i in val_idx]
validation_label = [label[i] for i in val_idx]
validation_masks = [masks[i] for i in val_idx]
validation_times = [time[i] for i in val_idx]
validation_top5_label = [top5_label[i] for i in val_idx]

test_features = [features[i] for i in test_idx]
test_aux = [aux[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]
test_label = [label[i] for i in test_idx]
test_masks = [masks[i] for i in test_idx]
test_times = [time[i] for i in test_idx]
test_top5_label = [top5_label[i] for i in test_idx]


# 检查构建的数据集大小
assert len(train_features) == len(train_aux) == len(train_label) == len(train_idx), "训练集大小不一致"
assert len(validation_features) == len(validation_aux) == len(validation_label) == len(val_idx), "验证集大小不一致"
assert len(test_features) == len(test_aux) == len(test_label) == len(test_idx), "测试集大小不一致"

print(f"最终训练集大小: {len(train_features)}")
print(f"最终验证集大小: {len(validation_features)}")
print(f"最终测试集大小: {len(test_features)}")

# 检查标签分布
train_label_counter = Counter(train_label)
val_label_counter = Counter(validation_label)
test_label_counter = Counter(test_label)

print(f"训练集标签分布: {dict(train_label_counter)}")
print(f"验证集标签分布: {dict(val_label_counter)}")
print(f"测试集标签分布: {dict(test_label_counter)}")

# 定义一个函数以保存数据
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# 保存训练集、验证集和测试集的特征、辅助特征和标签
save_data(train_features, f'../{DATASET_DIR}/train_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(train_aux, f'../{DATASET_DIR}/train_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(train_labels, f'../{DATASET_DIR}/train_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(train_masks, f'../{DATASET_DIR}/train_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(train_label, f'../{DATASET_DIR}/train_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(train_times, f'../{DATASET_DIR}/train_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(train_top5_label, f'../{DATASET_DIR}/train_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')


save_data(validation_features, f'../{DATASET_DIR}/validation_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(validation_aux, f'../{DATASET_DIR}/validation_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(validation_labels, f'../{DATASET_DIR}/validation_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(validation_masks, f'../{DATASET_DIR}/validation_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(validation_label, f'../{DATASET_DIR}/validation_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(validation_times, f'../{DATASET_DIR}/validation_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(validation_top5_label, f'../{DATASET_DIR}/validation_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')


save_data(test_features, f'../{DATASET_DIR}/test_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(test_aux, f'../{DATASET_DIR}/test_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(test_labels, f'../{DATASET_DIR}/test_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(test_masks, f'../{DATASET_DIR}/test_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(test_label, f'../{DATASET_DIR}/test_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(test_times, f'../{DATASET_DIR}/test_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
save_data(test_top5_label, f'../{DATASET_DIR}/test_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')



print("Data saved successfully.")