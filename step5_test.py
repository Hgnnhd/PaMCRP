from module.model import *
from config import *
from module.eval import *
from module.net import *
from torch.utils.data import DataLoader
seed_torch(2023)
num_classes = len(evaluation_times)
input_dim = DIMS
hidden_dim = 256
output_dim = num_classes
max_seq_length = 241
num_encoder_layers = 3
nhead = 16
test_features = load_data(f'../{DATASET_DIR}/test_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_aux = load_data(f'../{DATASET_DIR}/test_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_labels = load_data(f'../{DATASET_DIR}/test_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_masks = load_data(f'../{DATASET_DIR}/test_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_times = load_data(f'../{DATASET_DIR}/test_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_label = load_data(f'../{DATASET_DIR}/test_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_top5_label = load_data(f'../{DATASET_DIR}/test_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_dataset = CustomDataset_aux(test_features, test_aux, test_labels, test_masks, test_times, test_label, test_top5_label)
test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=collate_fn)

# Set your checkpoint path here
model_path = "best_model_ours.pth"  # edit to your checkpoint

model_name = model_path.split('_')[2]
print(model_name)
model = PaMCRP(input_dim, hidden_dim, output_dim, nhead, max_seq_length, num_encoder_layers, num_classes, model_type=model_name).to(device)
print(f"Testing {model.__class__.__name__} with {model_name} and {num_classes} classes")

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
results, roc_data = evaluate_top5(model, test_dataloader, test=False)
results_filename = save_training_results(results, model_path, './result/my_results', model_name)
print(f"Training results saved as {results_filename}")
print("best model saved as", model_path)
