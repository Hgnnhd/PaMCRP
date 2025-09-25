import runpy

# Step 1: Clean and select sample cohort
runpy.run_module("step1_cleansample")

# Step 2: Rank diagnoses by date and build labels/masks
runpy.run_module('step2_rankdate')

# Step 3: Map ICD codes and build indices/embeddings/dataset
runpy.run_module('step3_1_icdmapping')
runpy.run_module('step3_2_idx')
runpy.run_module('step3_3_embedding')
runpy.run_module('step3_4_dataset')



