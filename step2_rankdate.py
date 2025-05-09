import pickle
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple
from config import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_data(pattern: str) -> Tuple[pd.DataFrame]:
    try:
        data = pd.read_csv(f"../{DATASET_DIR}/step1_dataset_{pattern}.csv")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise


def load_aux_data() -> pd.DataFrame:
    """加载辅助数据"""
    try:
        return pd.read_csv(f"../{DATASET_DIR}/aux_data.csv")
    except FileNotFoundError as e:
        logging.error(f"Auxiliary data file not found: {e}")
        raise


def process_patient_diagnoses(data: pd.DataFrame) -> Dict:
    """处理患者诊断数据"""
    patient_diagnoses = {}
    for _, row in tqdm(data.iterrows()):
        participant_id = row["Participant ID"]
        icd10 = row["Diagnoses - ICD10"].split("|")

        for idx, item_icd in (enumerate(icd10)):
            date_x = row.get(f"Date of first in-patient diagnosis - ICD10 | Array {idx}", "2024/1/1")
            if pd.isna(date_x):
                # logging.warning(f"Missing date for participant {participant_id}, using default")
                date_x = "2099/1/1"

            if participant_id not in patient_diagnoses:
                patient_diagnoses[participant_id] = {}
            patient_diagnoses[participant_id][item_icd] = datetime.strptime(date_x, "%Y/%m/%d")

    return patient_diagnoses

def filter_recent_data(sorted_data, sorted_date, exclude_days=90):
    """排除最近一段时间的数据
    Args:
        sorted_data: 排序后的诊断代码列表
        sorted_date: 排序后的日期列表
        exclude_days: 需要排除的天数(默认90天,即3个月)
    Returns:
        filtered_data: 过滤后的诊断代码列表
        filtered_date: 过滤后的日期列表
    """
    if not sorted_date or not sorted_data:
        return sorted_data, sorted_date

    last_date = sorted_date[-1]
    exclude_date = last_date - timedelta(days=exclude_days)

    filtered_data = []
    filtered_date = []

    for code, date in zip(sorted_data, sorted_date):
        if date <= exclude_date:
            filtered_data.append(code)
            filtered_date.append(date)

    return filtered_data, filtered_date


def sort_and_filter_diagnoses(date_dict: Dict, data: pd.DataFrame, min_length=5, max_length=9999) -> Tuple[
    Dict, Dict, Dict]:
    # # 根据TARGET_PATTERN生成标签

    top5_label_dict = {
        "C34": 1,  # 肺癌
        "C50": 2,  # 乳腺癌
        "C18": 3,  # 结直肠癌
        "C19": 3,  # 结直肠癌
        "C20": 3,  # 结直肠癌
        "C61": 4,  # 前列腺癌
        "C16": 5 # 胃癌
    }

    new_date = {}
    new_dict = {}
    top5_label_list = []
    label_dict = []
    mask_dict = []
    # 评估点列表
    Endpoints = evaluation_times

    cancer_intervals = []
    year_intervals = {i: 0 for i in range(1, 21)}

    for participant_id, item_dict in tqdm(date_dict.items()):
        time_since_last = []
        time_to_last = []
        time_since_birth = []


        cancer_code = str(data[data["Participant ID"] == participant_id]["Type of cancer: ICD10 | Instance 0"].values[0])[:3]
        cancer_date = str(data[data["Participant ID"] == participant_id]["Date of cancer diagnosis | Instance 0"].values[0])
        if not cancer_date == "nan":
            cancer_date = datetime.strptime(cancer_date, "%Y/%m/%d")

        birth_date = data[data["Participant ID"] == participant_id]["Date of birth"].values[0]
        birth_date = datetime.strptime(birth_date, "%Y/%m/%d")

        sorted_items = sorted(item_dict.items(), key=lambda x: x[1])

        if len(sorted_items) < min_length:
            # logging.info(f"{participant_id}: insufficient diagnoses")
            continue

        def get_cancer_codes(cancer_code):
            cancer_codes = {
                'C18': ['C18', 'C19', 'C20', 'Z85.0', 'D01.0', 'D01.1', 'D01.2'],  # 结直肠癌
                'C19': ['C18', 'C19', 'C20', 'Z85.0', 'D01.0', 'D01.1', 'D01.2'],  # 结直肠癌
                'C20': ['C18', 'C19', 'C20', 'Z85.0', 'D01.0', 'D01.1', 'D01.2'],  # 结直肠癌
                'C50': ['C50', 'Z85.3', 'D05'],  # 乳腺癌
                'C34': ['C34', 'Z85.1', 'D02.2'],  # 肺癌
                'C61': ['C61', 'Z85.4', 'D07.5'],  # 前列腺癌
                'C16': ['C16', 'Z85.0', 'D00.2'],  # 胃癌
                # 可以根据需要添加更多癌症类型
            }
            return cancer_codes.get(cancer_code, [cancer_code])

        if not cancer_code == 'nan':
            relevant_codes = get_cancer_codes(cancer_code)
            cancer_index = next((i for i, items in enumerate(sorted_items)
                                 if any(items[0].startswith(code) for code in relevant_codes)), None)

            if cancer_index == None:  # 说明没有找到目标癌症诊断
                # 只要满足时间小于癌症诊断时间即可
                cancer_before = cancer_date - timedelta(days=1)
                sorted_items = [item for item in sorted_items if item[1] <= cancer_before]
            else:
                cancer_date = sorted_items[cancer_index][1]
                sorted_items = sorted_items[:cancer_index]
                cancer_before = cancer_date - timedelta(days=1)
                sorted_items = [item for item in sorted_items if item[1] <= cancer_before]


            if len(sorted_items) < 1:
                # logging.info(f"{participant_id}: insufficient diagnoses")
                continue

            last_diagnosis_date = sorted_items[-1][1]
            interval = (cancer_date - last_diagnosis_date).days
            cancer_intervals.append(interval)

            # 统计年份间隔
            year_interval = interval // 365 + 1
            if year_interval <= 20:
                year_intervals[year_interval] += 1
        else:
            last_date = sorted_items[-1][1]
            n_years_ago = last_date - timedelta(days=EXCLUDE_DAY_Normal)
            sorted_items = [item for item in sorted_items if item[1] <= n_years_ago]

        sorted_data = [item.split(" ")[0] for item, _ in sorted_items]
        sorted_date = [date for _, date in sorted_items]

        if len(sorted_data) < 1:
            # logging.info(f"{participant_id}: insufficient diagnoses")
            continue

        end_date = sorted_date[-1]
        # sorted_data, sorted_date = filter_recent_data(sorted_data, sorted_date, exclude_days=EXCLUDE_DAY)

        if len(sorted_data) < 1:
            # logging.info(f"{participant_id}: insufficient diagnoses")
            continue

       # 计算时间间隔
        for i, (item, date) in enumerate(zip(sorted_data, sorted_date)):
            if i == 0:
                time_since_last.append(0)  # 第一个疾病，与上一个疾病的时间间隔为0
            else:
                time_since_last.append((date - sorted_date[i - 1]).days)

            time_to_last.append((sorted_date[-1] - date).days)
            time_since_birth.append((date - birth_date).days)

        new_date[participant_id] = [time_since_last,time_to_last,time_since_birth]
        new_dict[participant_id] = sorted_data

        follow_days = (cancer_date - end_date).days if not cancer_code=='nan' else (last_date - end_date).days

        num_time_steps, max_time = len(Endpoints), max(Endpoints)
        y = follow_days < max_time and pd.isna(cancer_code)
        y_seq = np.zeros(num_time_steps)

        if follow_days < max_time:
            time_at_event = min([i for i, mo in enumerate(Endpoints) if follow_days < mo])
        else:
            time_at_event = num_time_steps - 1

        if y:
            y_seq[time_at_event] = 1

        y_mask = np.array([1] * (time_at_event + 1) + [0] * (num_time_steps - (time_at_event + 1)))

        if cancer_code in top5_label_dict:
            top5_label = top5_label_dict[cancer_code]
        else:
            top5_label = 0

        top5_label_list.append(top5_label)

        label_dict.append(y_seq)
        mask_dict.append(y_mask)

    label_dict = np.stack(label_dict, axis=0)
    mask_dict = np.stack(mask_dict, axis=0)

    with open(f"../{DATASET_DIR}/time_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}.pkl", "wb") as f:
            pickle.dump(new_date, f)

    # 计算癌症间隔的统计数据
    if cancer_intervals:
        max_interval = max(cancer_intervals)
        min_interval = min(cancer_intervals)
        avg_interval = sum(cancer_intervals) / len(cancer_intervals)
        logging.info(
            f"Cancer intervals - Max: {max_interval} days, Min: {min_interval} days, Average: {avg_interval:.2f} days")
    else:
        logging.info("No cancer patients found in the dataset")

    plot_year_intervals(year_intervals)

    label_dict = np.stack(label_dict,axis=0)
    mask_dict = np.stack(mask_dict,axis=0)
    top5_label_list = np.stack(top5_label_list,axis=0)

    return new_dict, label_dict, mask_dict, top5_label_list

def plot_year_intervals(year_intervals):
    years = list(year_intervals.keys())
    counts = list(year_intervals.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(years, counts, color='skyblue', edgecolor='navy')

    plt.title('Distribution of Time Intervals Before Cancer Diagnosis', fontsize=16)
    plt.xlabel('Years before Cancer Diagnosis', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.xticks(years)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')

    # 优化布局
    plt.tight_layout()

    # 保存图表
    plt.savefig('save/cancer_interval_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_data(data: Dict, filename: str):
    """保存数据到文件"""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_or_process_data(pattern: str, filename: str):
    """加载现有数据或处理新数据"""
    # if os.path.exists(filename):
    #     logging.info(f"Loading existing data from {filename}")
    #     with open(filename, "rb") as f:
    #         return pickle.load(f)
    # else:
    logging.info("Processing new data")
    data = load_data(pattern)
    data = data.sort_values(by="Participant ID").reset_index(drop=True)
    patient_diagnoses = process_patient_diagnoses(data)
    save_data(patient_diagnoses, filename)
    return patient_diagnoses


def main():

    date_dict_file = f"../{DATASET_DIR}/date_dict_{'_'.join(TARGET_PATTERN)}.pkl"
    # 加载或处理数据
    patient_diagnoses = load_or_process_data('_'.join(TARGET_PATTERN), date_dict_file)
    # 加载其他必要的数据
    data = load_data('_'.join(TARGET_PATTERN))
    # 处理和保存结果
    new_dict, label_dict, mask_dict, top5_label_dict = sort_and_filter_diagnoses(patient_diagnoses, data, min_length=Min_length)
    save_data(new_dict, f"../{DATASET_DIR}/icd10_order_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}.pkl")
    save_data(label_dict, f"../{DATASET_DIR}/label_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl")
    save_data(mask_dict, f"../{DATASET_DIR}/mask_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl")
    save_data(top5_label_dict, f"../{DATASET_DIR}/top5_label_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl")
    ##报告人数
    logging.info(f"Number of participants: {len(new_dict)}")
    logging.info("Data processing completed successfully.")

main()











