import time
from config import *
from collections import Counter
from module.net import *
from module.eval import *
import torch.optim as optim
from module.utils import *

seed_torch(2023)
epochs = 100
batch_size = 1024
num_classes = len(evaluation_times)
lr = 5e-4
input_dim = DIMS
hidden_dim = 256
output_dim = num_classes
max_seq_length = 241
num_encoder_layers = 3
nhead = 16
patience = 10
counter = 10

# Step 1: Load train/val/test features, aux, times, labels, masks
train_features = load_data(f'../{DATASET_DIR}/train_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_features = load_data(f'../{DATASET_DIR}/validation_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_features = load_data(f'../{DATASET_DIR}/test_features_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_aux = load_data(f'../{DATASET_DIR}/train_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_aux = load_data(f'../{DATASET_DIR}/validation_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_aux = load_data(f'../{DATASET_DIR}/test_aux_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_times = load_data(f'../{DATASET_DIR}/train_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_times = load_data(f'../{DATASET_DIR}/validation_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_times = load_data(f'../{DATASET_DIR}/test_times_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_labels = load_data(f'../{DATASET_DIR}/train_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_labels = load_data(f'../{DATASET_DIR}/validation_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_labels = load_data(f'../{DATASET_DIR}/test_labels_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_label = load_data(f'../{DATASET_DIR}/train_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_label = load_data(f'../{DATASET_DIR}/validation_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_label = load_data(f'../{DATASET_DIR}/test_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_masks = load_data(f'../{DATASET_DIR}/train_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_masks = load_data(f'../{DATASET_DIR}/validation_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_masks = load_data(f'../{DATASET_DIR}/test_masks_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

train_top5_label = load_data(f'../{DATASET_DIR}/train_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
validation_top5_label = load_data(f'../{DATASET_DIR}/validation_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')
test_top5_label = load_data(f'../{DATASET_DIR}/test_top5_label_{EXCLUDE_DAY}_{"_".join(TARGET_PATTERN)}_{LEVEL}_aux.pkl')

print("\nDataset sizes...")
print(f"Train: {len(train_features)}")
print(f"Val: {len(validation_features)}")
print(f"Test: {len(test_features)}")

print("\nCancer label distribution...")
print(f"Train: {dict(Counter(train_label))}")
print(f"Val: {dict(Counter(validation_label))}")
print(f"Test: {dict(Counter(test_label))}")

print("\nTop-5 label distribution...")
print(f"Train: {dict(sorted(Counter(train_top5_label).items()))}")
print(f"Val: {dict(sorted(Counter(validation_top5_label).items()))}")
print(f"Test: {dict(sorted(Counter(test_top5_label).items()))}")
print("Data loaded successfully.")


def train(model, train_loader, optimizer, patero):
    model.train()
    running_loss = 0.0
    for step, (inputs, aux, labels, masks, time, gold, top5_gold) in enumerate(train_loader):
        # Step 2: Move batch to device
        inputs = inputs.long().to(device)
        aux = aux.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        time = time.to(device)
        gold = gold.long().to(device)
        top5_gold = top5_gold.long().to(device)

        # Step 3: Forward pass and compute losses
        outputs, cancer_pred, top5_cancer_pred, proxy_loss = model(inputs, time, aux, top5_gold)
        loss_main = 0
        loss_balance = []
        loss_balance.append(proxy_loss)

        for i in range(5):
            is_current_class_or_normal = (gold == (i + 1)) | (gold == 0)
            current_outputs = outputs[i][is_current_class_or_normal]
            current_labels = labels[is_current_class_or_normal]
            current_masks = masks[is_current_class_or_normal]
            if current_outputs.size(0) == 0:
                continue
            ce_loss = F.cross_entropy(current_outputs, current_labels, reduction='none')
            ce_loss = ce_loss * current_masks
            ce_loss = ce_loss.sum() / current_masks.sum()

            loss_main += ce_loss

        loss_t = F.cross_entropy(top5_cancer_pred, top5_gold)
        loss_balance.append(loss_main)
        loss_balance.append(loss_t)

        # Use Patero (Pareto via MGDA) to balance multiple objectives and backprop
        _, weighted_loss = patero.backward(loss_balance)
        optimizer.step()

        running_loss += weighted_loss.item()

    return running_loss / len(train_loader)


train_dataset = CustomDataset_aux(train_features, train_aux, train_labels, train_masks, train_times, train_label,train_top5_label)
validation_dataset = CustomDataset_aux(validation_features, validation_aux, validation_labels, validation_masks, validation_times, validation_label,validation_top5_label)
test_dataset = CustomDataset_aux(test_features, test_aux, test_labels, test_masks, test_times, test_label, test_top5_label)
# Step 4: Wrap datasets with DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
# Step 5: Define model and optimizer
model = PaMCRP(input_dim, hidden_dim, output_dim, nhead, max_seq_length, num_encoder_layers, num_classes, model_type).to(device)
# Initialize Patero (implemented via MGDA) for 3 objective terms: proxy, main, top-5
patero = Patero(model, task_num=3, device=str(device))
params = list(model.parameters())
optimizer = optim.Adam(params, lr=lr)
# Step 6: Train with early stopping; track best model by Avg AUC
print(f"Training {model.__class__.__name__} with {model_type} and {num_classes} classes")
pbar = tqdm(range(epochs))
best_val_auc = 0.0
best_epoch = 0
for epoch in pbar:
    a = time.time()
    loss = train(model, train_dataloader, optimizer, patero)
    results, roc_data = evaluate_top5(model, test_dataloader)
    pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
    train_time = time.time() - a
    # Calculate average AUC and CI for each category
    avg_auc_categories = {}
    avg_auc_ci_categories = {}
    for category in range(1, 6):
        avg_auc = sum(results.get(f'Time_{t}_{category}_AUC', 0) for t in evaluation_times) / len(
            evaluation_times)
        avg_auc_categories[f'Avg_AUC_{category}'] = avg_auc
        # Calculate average CI
        ci_lowers = [results.get(f'Time_{t}_{category}_AUC_CI', (0, 0))[0] for t in evaluation_times]
        ci_uppers = [results.get(f'Time_{t}_{category}_AUC_CI', (0, 0))[1] for t in evaluation_times]
        avg_ci_lower = sum(ci_lowers) / len(evaluation_times)
        avg_ci_upper = sum(ci_uppers) / len(evaluation_times)
        avg_auc_ci_categories[f'Avg_AUC_CI_{category}'] = (avg_ci_lower, avg_ci_upper)
    # Update progress bar suffix
    postfix = {
        "best_epoch": f"{best_epoch:.1f}",
        "best_auc": f"{best_val_auc:.4f}",
        'Loss': f"{loss:.4f}",
        'Time': f"{train_time:.2f}s",
    }
    avg_auc_roc = sum(avg_auc_categories.values()) / len(avg_auc_categories)
    postfix['Avg AUC'] = f"{avg_auc_roc:.4f}"
    for cat in range(1, 6):
        auc = avg_auc_categories[f'Avg_AUC_{cat}']
        ci_lower, ci_upper = avg_auc_ci_categories[f'Avg_AUC_CI_{cat}']
        postfix[f'AUC_{cat}'] = f"{auc:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
    # Add C-index and CI to postfix
    for category in range(1, 6):
        c_index = results.get(f"C_index_{category}", "N/A")
        c_index_ci = results.get(f"C_index_CI_{category}", ("N/A", "N/A"))
        if isinstance(c_index, float) and isinstance(c_index_ci[0], float) and isinstance(c_index_ci[1], float):
            postfix[f"C-index_{category}"] = f"{c_index:.4f} ({c_index_ci[0]:.4f}-{c_index_ci[1]:.4f})"
    pbar.set_postfix(postfix)
    # Save the best performing model
    if avg_auc_roc > best_val_auc:
        best_val_auc = avg_auc_roc
        best_model_state_dict = model.state_dict()
        # Add timestamp
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f'../results/best_model_{model_type}_{current_time}.pth'
        torch.save(best_model_state_dict, name)
        best_epoch = epoch + 1
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        break
# Step 8: Final evaluation with best checkpoint
model.load_state_dict(torch.load(name))
results, roc_data = evaluate_top5(model, test_dataloader, test=True)

others = f"{EXCLUDE_DAY}"
print_results(results)
results_filename = save_training_results(results, name, './result/my_results', others)
print(f"Training results saved as {results_filename}")
print("best model saved as", name)
