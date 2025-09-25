PaMCRP is a pipeline for passive multi-cancer risk prediction from longitudinal EHR. It builds patient disease trajectories from ICD-10 codes and temporal features, constructs datasets, and trains a transformer-based model with auxiliary demographic/behavioral features.

This README provides the minimal system requirements, recommended software versions, data preparation, and a step-by-step run guide.

Overview
- Inputs: EHR table with ICD-10 sequences, diagnosis dates, demographics/auxiliaries.
- Outputs: Prepared datasets, split indices, trained checkpoint(s), and evaluation metrics/plots.
- Target cancers: Top 5 incidence cancers (Lung C34, Breast C50, Colorectal C18-C20, Prostate C61, Stomach C16).

System Requirements
- OS: Linux (Ubuntu 20.04/22.04) or similar; macOS works for CPU; Windows WSL2 recommended.
- CPU: x86_64; 16 GB RAM recommended.
- GPU: Optional but recommended. NVIDIA GPU with CUDA 11.x/12.x and >=8 GB VRAM.
- Disk: ~10 GB free for datasets and artifacts (depends on your data size).

Software Versions (tested)
- Python: 3.9-3.11 (recommend 3.10)
- PyTorch: 2.0-2.3 (with matching CUDA build if using GPU)
- CUDA Toolkit: 11.8 or 12.1 (optional if using GPU)
- Others: pandas>=1.5, numpy>=1.23, scikit-learn>=1.1, tqdm, matplotlib, gensim

Quick Environment Setup
1) Create environment (example with conda):
   conda create -n pamcrp python=3.10 -y
   conda activate pamcrp

2) Install PyTorch (choose the command for your CUDA):
   # CUDA 11.8 example
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # CPU-only example
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

3) Install Python deps:
   pip install pandas numpy scikit-learn tqdm matplotlib gensim

Repository Structure (key files)
- config.py: Global settings (device, time horizons, dataset path, etc.).
- run.py: Orchestrates steps 1-3 preprocessing.
- step1_cleansample.py: Build cohort subset and save step-1 CSV.
- step2_rankdate.py: Sort diagnoses, compute temporal features, labels/masks; save dictionaries and plots.
- step3_1_icdmapping.py: ICD-10 code utilities (chapters/categories mapping, helpers).
- step3_2_idx.py: Stratified train/val/test split; save indices.
- step3_3_embedding.py: Build per-patient tuples with aux and time features.
- step3_4_dataset.py: Materialize split datasets and persist artifacts.
- step4_train.py: Train model; save best checkpoint and results.
- step5_test.py: Load checkpoint and evaluate on test set.

Data Preparation
Place your raw EHR CSVs in a sibling folder to the repo root:
- ../data/dataset.csv - Main table with at least the following columns:
  - Participant ID
  - Diagnoses - ICD10 (pipe-separated list; e.g., C50.9|I10|...)
  - Date of first in-patient diagnosis - ICD10 | Array {k} (one column per ICD index, format YYYY/MM/DD)
  - Type of cancer: ICD10
  - Date of cancer diagnosis (YYYY/MM/DD)
  - Date of birth (YYYY/MM/DD)
  - Sex, Age (derived from time_since_birth)

- ../dataset/ (output directory, created if missing). Some steps also read from this folder.
- Optional: ../dataset/aux_data.csv if you intend to extend auxiliary features (see step2_rankdate.py placeholder).

Important Configs (config.py)
- gpu_id: GPU index when CUDA is available.
- evaluation_times: Time horizons in days (e.g., 3, 6, 12, 36, 60 months).
- LEVEL: ICD granularity level (used by downstream mapping).
- EXCLUDE_DAY, EXCLUDE_DAY_Normal: Temporal exclusion windows.
- DATASET_DIR: Output folder name (default "dataset").
- Min_length: Minimum number of diagnoses required per patient.
- model_type: Model variant flag (e.g., "ours").
- DIMS: Embedding size.
- TARGET_PATTERN: List of ICD "C" codes to focus on (e.g., ["34", "50"]) - define in your environment or inject into config if needed.

Running the Pipeline
0) From repo root: cd PaMCRP

1) Preprocess (steps 1-3) - build cohort and artifacts:
   python run.py

   This runs:
   - step1_cleansample.py -> ../dataset/step1_dataset_<TARGET>.csv
   - step2_rankdate.py -> time/label/mask/top5_label pickles, plots under ./save/
   - step3_1_icdmapping.py, step3_2_idx.py, step3_3_embedding.py, step3_4_dataset.py

2) Train the model:
   python step4_train.py

   Artifacts:
   - ../results/best_model_<model_type>_<timestamp>.pth
   - ./result/my_results/* (metrics, AUCs, C-index)

3) Test/evaluate a saved checkpoint:
   python step5_test.py

   Edit model_paths in step5_test.py to point to your checkpoint path if needed.

Outputs
- ../dataset/: All intermediate pickles and split artifacts (features/aux/labels/masks/times and indices).
- ./save/: Plots such as cancer_interval_distribution.png.
- ../results/: Trained model checkpoints.
- ./result/my_results/: Aggregated evaluation results.

Notes and Tips
- GPU selection: config.py auto-selects CUDA if available (uses gpu_id). For CPU-only, PyTorch will fall back automatically.
- Date formats: Code assumes YYYY/MM/DD for date columns in the input CSV; adjust readers if your format differs.
- Ethnicity mapping: step3_3_embedding.py maps several coded values to coarse groups; unknowns default to 4 (Other).
- Reproducibility: seed_torch(2023) is used in training/eval utilities.

Troubleshooting
- Missing module errors (e.g., module.*): Ensure the "module" package with model/net/utils/eval files exists on PYTHONPATH if not included here.
- Data columns not found: Verify column names match exactly (case and spacing matter). Adjust code where necessary.
- CUDA errors: Install a PyTorch build compatible with your CUDA driver, or switch to CPU-only build.
- Memory issues: Reduce batch_size in step4_train.py, shorten sequences, or filter smaller cohorts.
