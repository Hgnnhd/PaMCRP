import pickle
from collections import Counter
from config import *

# Step 1: Load prepared tuples and related labels/masks
dataset = pickle.load(open(f"../{DATASET_DIR}/patient_dataset_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}_aux.pkl", "rb"))
features, aux, time, label = zip(*dataset)

labels = pickle.load(open(f"../{DATASET_DIR}/label_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb"))
masks = pickle.load(open(f"../{DATASET_DIR}/mask_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb"))
top5_label = pickle.load(open(f"../{DATASET_DIR}/top5_label_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb"))

# Step 2: Load index splits
with open(f"../{DATASET_DIR}/train_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    train_idx = pickle.load(f)
with open(f"../{DATASET_DIR}/val_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    val_idx = pickle.load(f)
with open(f"../{DATASET_DIR}/test_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    test_idx = pickle.load(f)

# Step 3: Basic sanity checks
train_set = set(train_idx)
val_set = set(val_idx)
test_set = set(test_idx)
all_indices = set(range(len(features)))
used_indices = train_set | val_set | test_set

print(f"Train size: {len(train_set)}")
print(f"Val size: {len(val_set)}")
print(f"Test size: {len(test_set)}")
print(f"Used indices: {len(used_indices)}")
print(f"Total features: {len(features)}")

# Step 4: Build split datasets
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

# Step 5: Assert sizes and show distributions
assert len(train_features) == len(train_aux) == len(train_label) == len(train_idx), "Train split size mismatch"
assert len(validation_features) == len(validation_aux) == len(validation_label) == len(val_idx), "Val split size mismatch"
assert len(test_features) == len(test_aux) == len(test_label) == len(test_idx), "Test split size mismatch"

print(f"Final train size: {len(train_features)}")
print(f"Final val size: {len(validation_features)}")
print(f"Final test size: {len(test_features)}")

train_label_counter = Counter(train_label)
val_label_counter = Counter(validation_label)
test_label_counter = Counter(test_label)
print(f"Train label distribution: {dict(train_label_counter)}")
print(f"Val label distribution: {dict(val_label_counter)}")
print(f"Test label distribution: {dict(test_label_counter)}")

# Step 6: Save split artifacts
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

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
