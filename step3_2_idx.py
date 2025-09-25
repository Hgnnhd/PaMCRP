import pickle
from config import *
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Step 1: Load patient disease id sequences and labels
with open(f"../{DATASET_DIR}/patient_dis2id_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as pickle_file:
    disease_data = pickle.load(pickle_file)

top5_label = pickle.load(open(f"../{DATASET_DIR}/top5_label_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb"))

disease_data = list(disease_data.values())
labels = np.array([top5_label[i] for i in range(len(disease_data))])

# Step 2: Stratified split into train/val/test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, temp_index in sss.split(disease_data, labels):
    pass

sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in sss_temp.split([disease_data[i] for i in temp_index], labels[temp_index]):
    val_idx = temp_index[val_index]
    test_idx = temp_index[test_index]

train_idx = train_index

# Step 3: Persist split indices
with open(f"../{DATASET_DIR}/train_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "wb") as pickle_file:
    pickle.dump(train_idx, pickle_file)

with open(f"../{DATASET_DIR}/val_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "wb") as pickle_file:
    pickle.dump(val_idx, pickle_file)

with open(f"../{DATASET_DIR}/test_idx_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "wb") as pickle_file:
    pickle.dump(test_idx, pickle_file)

# Step 4: Save only training sequences for id mapping convenience
disease_data = [disease_data[i] for i in train_idx]
with open(f"../{DATASET_DIR}/id_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "wb") as pickle_file:
    pickle.dump(disease_data, pickle_file)

print('Index split and save finished!')


