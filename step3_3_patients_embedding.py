import pickle
import pandas as pd
from config import *

with open(f"../{DATASET_DIR}/dis2id_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    dis_dict = pickle.load(f)

with open(f"../{DATASET_DIR}/patient_dis2id_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "rb") as f:
    dataset = pickle.load(f)

with open(f"../{DATASET_DIR}/time_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}.pkl", "rb") as f:
    date_dict = pickle.load(f)

data = pd.read_csv("../dataset/step1_dataset_{}.csv".format('_'.join(TARGET_PATTERN)))
ethnic_mapping = {
    # White Groups = 0
    0: 0,  # British -> White
    1: 0,  # Any other white background -> White
    4: 0,  # Irish -> White
    18: 0,  # White -> White

    # Asian Groups = 1
    3: 1,  # Indian -> Asian
    7: 1,  # Chinese -> Asian
    8: 1,  # Any other Asian background -> Asian
    11: 1,  # Pakistani -> Asian
    15: 1,  # Asian or Asian British -> Asian
    16: 1,  # Bangladeshi -> Asian

    # Black Groups = 2
    5: 2,  # African -> Black
    6: 2,  # Caribbean -> Black
    20: 2,  # Any other Black background -> Black
    22: 2,  # Black or Black British -> Black

    # Mixed Groups = 3
    9: 3,  # White and Asian -> Mixed
    10: 3,  # Any other mixed background -> Mixed
    14: 3,  # White and Black African -> Mixed
    17: 3,  # White and Black Caribbean -> Mixed
    19: 3,  # Mixed -> Mixed

    # Others = 4
    2: 4,  # Other ethnic group -> Other
    12: 4,  # Prefer not to answer -> Other
    13: 4,  # NaN -> Other
    21: 4  # Do not know -> Other
}

# 对ethnic列进行映射
data['Ethnic background | Instance 0'] = data['Ethnic background | Instance 0'].map(lambda x: ethnic_mapping.get(x, 4))

# 定义要包含的额外特征
extra_features = [
    # 'Townsend deprivation index at recruitment',
    # "Father's age at death | Instance 0",
    # "Mother's age at death | Instance 0",
    # 'Duration of moderate activity | Instance 0',
    # 'MET minutes per week for walking | Instance 0',
    # 'Water intake | Instance 0',
    # 'Fresh fruit intake | Instance 0',
    # 'Sleep duration | Instance 0',
    # 'Getting up in morning | Instance 0',
    "Sex",
    'Ethnic background | Instance 0',
    'Ever smoked | Instance 0',
    'Alcohol drinker status | Instance 0',

    "Standing height | Instance 0",
    "Body mass index (BMI) | Instance 0",

    "Age high blood pressure diagnosed | Instance 0",
    "Ever had diabetes (Type I or Type II)",

    "Illnesses of father | Instance 0",
    "Illnesses of mother | Instance 0",
    "Illnesses of siblings | Instance 0",

]


def transform_extra_features(aux_features, extra_features):
    transformed = []

    # 定义已知的疾病组
    group1_diseases = ["Heart disease", "Stroke", "High blood pressure",
                       "Chronic bronchitis/emphysema",
                       "Alzheimer's disease/dementia", "Diabetes"]
    group2_diseases = ["Parkinson's disease", "Severe Depression",
                       "Lung cancer", "Bowel cancer", "Prostate cancer",
                       "Breast cancer"]
    all_diseases = set(group1_diseases + group2_diseases)

    for feat, value in zip(extra_features, aux_features):
        if feat == "Sex":
            transformed.append(value)

        elif feat == "Ethnic background | Instance 0":
            # 不处理，直接传递原值
            transformed.append(value)

        elif feat in ["Standing height | Instance 0", "Body mass index (BMI) | Instance 0"]:
            # 转换为整数
            if pd.isna(value):
                transformed.append(-1)
            else:
                transformed.append(int(float(value)))

        elif feat == "Age high blood pressure diagnosed | Instance 0":
            # 存在数字为1，其他为0
            if pd.isna(value):
                transformed.append(0)
            else:
                try:
                    float(value)  # 尝试转换为数字
                    transformed.append(1)
                except ValueError:
                    transformed.append(0)

        elif feat == "Ever had diabetes (Type I or Type II)":
            # Yes=1，其他为0
            transformed.append(1 if value == "Yes" else 0)

        elif feat == 'Sleeplessness / insomnia | Instance 0':
            transformed.append(value)

        elif feat == 'Ever smoked | Instance 0':
            transformed.append(value)

        elif feat == 'Alcohol drinker status | Instance 0':
            transformed.append(value)

    if "Illnesses of father | Instance 0" in extra_features:
        # 处理家族疾病历史
        family_diseases = set()
        for feat, value in zip(extra_features, aux_features):
            if feat in ["Illnesses of father | Instance 0",
                        "Illnesses of mother | Instance 0",
                        "Illnesses of siblings | Instance 0"]:
                if not pd.isna(value):
                    diseases = [d.strip() for d in str(value).split('|')]
                    family_diseases.update(diseases)

        # 为每种已知疾病添加二元特征
        for disease in all_diseases:
            transformed.append(1 if disease in family_diseases else 0)

    return transformed


all_data = []
for p_id, dis in dataset.items():
    print("id :", p_id)
    patient_data = data[data["Participant ID"] == p_id]

    if patient_data.empty:
        print(f"Warning: No data found for patient ID {p_id}")
        continue

    temp_label = patient_data["Type of cancer: ICD10 | Instance 0"].values[0]
    mapped_label = 0 if pd.isna(temp_label) else 1

    # 获取额外特征
    aux_features = []
    for feature in extra_features:
        value = patient_data[feature].values[0]
        aux_features.append(value)

    transformed_features = transform_extra_features(aux_features, extra_features)
    time_features = date_dict[p_id]
    # 将疾病特征、辅助特征和标签分开保存
    all_data.append((dis, transformed_features, time_features, mapped_label))

with open(f"../{DATASET_DIR}/patient_dataset_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}_aux.pkl",
          "wb") as pickle_file:
    pickle.dump(all_data, pickle_file)

print("done")

