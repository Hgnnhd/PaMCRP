import pandas as pd
import re
from config import *


import pandas as pd
import re
data = pd.read_csv("../data/dataset.csv")
has_icd10 = data["Diagnoses - ICD10"].notna()
has_icd10_only = has_icd10
has_cancer = data["Type of cancer: ICD10 | Instance 0"].notna()

if not TARGET_PATTERN:
    target_cancer = has_icd10_only & has_cancer
else:
    target_pattern = "^C(" + "|".join(TARGET_PATTERN) + r")\b"
    target_cancer = has_icd10_only & has_cancer & data["Type of cancer: ICD10 | Instance 0"].str.match(target_pattern, na=False)


normal_patients = has_icd10_only & ~has_cancer

if TARGET_PATTERN == ["50"]:
    normal_patients = normal_patients & (data["Sex"] == 1)  #
elif TARGET_PATTERN == ["61"]:
    normal_patients = normal_patients & (data["Sex"] == 0)  #


if sum(normal_patients) < sum(target_cancer):
    print(f"Warning: Not enough normal patients ({sum(normal_patients)}) to match target cancer patients ({sum(target_cancer)}).")
    print("Using all available normal patients.")
    balanced_normal_patients = normal_patients
else:
    balanced_normal_patients = normal_patients


print(f"Original normal patients: {sum(normal_patients)}")
print(f"Balanced normal patients: {sum(balanced_normal_patients)}")
print(f"Target cancer patients: {sum(target_cancer)}")

target_indices = target_cancer | balanced_normal_patients
step1_dataset = data.loc[target_indices]

print("Target cancer patients:", sum(target_cancer))
print("Normal patients:", sum(balanced_normal_patients))
print("Total selected patients:", sum(target_indices))

output_filename = f"../dataset/step1_dataset_{'_'.join(TARGET_PATTERN)}.csv"
step1_dataset.to_csv(output_filename, index=False)
print(f"Dataset saved as {output_filename}")








