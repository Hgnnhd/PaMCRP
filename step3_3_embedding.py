import pickle
import pandas as pd
from config import *

# Step 1: Load patient sequences and time features
with open(f"../{DATASET_DIR}/patient_dis2id_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    dataset = pickle.load(f)

with open(f"../{DATASET_DIR}/time_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}.pkl", "rb") as f:
    date_dict = pickle.load(f)

# Step 2: Load cohort table
data = pd.read_csv("../dataset/step1_dataset_{}.csv".format('_'.join(TARGET_PATTERN)))

# Step 3: Select auxiliary features to include
extra_features = ["Sex"]


# Step 5: Build per-patient tuples and save
all_data = []
for p_id, dis in dataset.items():
    print("id :", p_id)
    patient_data = data[data["Participant ID"] == p_id]

    if patient_data.empty:
        print(f"Warning: No data found for patient ID {p_id}")
        continue

    temp_label = patient_data["Type of cancer: ICD10"].values[0]
    mapped_label = 0 if pd.isna(temp_label) else 1

    # Aux features: keep only Sex and Age (derived from time_since_birth)
    sex_value = patient_data["Sex"].values[0]
    time_features = date_dict[p_id]
    if isinstance(time_features, (list, tuple)) and len(time_features) == 3:
        time_since_birth = time_features[2]
        age_years = int(max(time_since_birth) / 365) if len(time_since_birth) > 0 else -1
    else:
        age_years = -1
    transformed_features = [sex_value, age_years]
    all_data.append((dis, transformed_features, time_features, mapped_label))

with open(f"../{DATASET_DIR}/patient_dataset_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}_aux.pkl",
          "wb") as pickle_file:
    pickle.dump(all_data, pickle_file)

print("Embedding dataset prepared.")

