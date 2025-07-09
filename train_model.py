import pandas as pd
from collections import Counter

from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.GRU.BiGRU_LSE import LSE_Bidirectional_GRU
from models.LSTM.BiLSTM_LSE import LSE_Bidirectional_LSTM
from data_preparations.csv_sequence_dataset import CSVSequenceDataset
from data_preparations.windowed_csv_dataset import WindowedCSVDataset

DatasetClass = CSVSequenceDataset

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    lengths, sort_idx = lengths.sort(descending=True)
    sequences_padded = sequences_padded[sort_idx]
    labels = labels[sort_idx]
    return sequences_padded, labels, lengths

def focal_loss(inputs, targets, alpha=None, gamma=3.5):
    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)
    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
    pt = torch.sum(prob * targets_one_hot, dim=-1)

    if alpha is not None:
        at = alpha[targets]
    else:
        at = 1.0

    loss = -at * (1 - pt) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)
    return loss.mean()

def get_class_weights(dataset, device):
    labels = [label for _, label, _ in dataset]
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    num_classes = len(dataset.class_to_idx)
    weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    return class_weights

def load_file_list(csv_path):
    return pd.read_csv(csv_path).iloc[:, 0].dropna().tolist()

def load_dataset(dataset_class, root_dir, drop_columns, file_list, **kwargs):
    return dataset_class(
        root_dir=root_dir,
        drop_columns=drop_columns,
        file_list=file_list,
        **kwargs
    )


# -------- Datasets y Dataloaders --------

train_files = load_file_list('/home/gerardo/LSE_HEALTH/LSE_TFG/train_test_val_split/train_weigthed_samples.csv')
val_files   = load_file_list('/home/gerardo/LSE_HEALTH/LSE_TFG/train_test_val_split/val_weigthed_samples.csv')
test_files  = load_file_list('/home/gerardo/LSE_HEALTH/LSE_TFG/train_test_val_split/test_weigthed_samples.csv')

common_args = {
    'drop_columns': ['frame'],
    'file_list': train_files
}

extra_args = {
    'target_frames': 11,
    'stride': 5
} if DatasetClass == WindowedCSVDataset else {}

train_dataset = load_dataset(DatasetClass, '/home/gerardo/FEATURES_POSE_DATASET', **common_args, **extra_args)

common_args['file_list'] = val_files
val_dataset = load_dataset(DatasetClass, '/home/gerardo/FEATURES_POSE_DATASET', **common_args, **extra_args)

common_args['file_list'] = test_files
test_dataset = load_dataset(DatasetClass, '/home/gerardo/FEATURES_POSE_DATASET', **common_args, **extra_args)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# -------- Hiperparámetros --------

sample_sequence, _, _ = train_dataset[0]
input_size = sample_sequence.shape[1]
num_classes = len(train_dataset.class_to_idx)

hidden_size = sample_sequence.shape[1] // 3
num_layers = 2
dropout = 0.5
bidirectional = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LSE_Bidirectional_LSTM(input_size, hidden_size, num_layers, num_classes, dropout=dropout, bidirectional=bidirectional)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -------- Calcular pesos de clase --------
class_weights = get_class_weights(train_dataset, device)
print("Class Weights:", class_weights)

#criterion = nn.CrossEntropyLoss(weight=class_weights)
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_without_improvement = 0

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.1,
    patience=3,
    verbose=True
)

# -------- Entrenamiento --------

num_epochs = 200

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
prev_lr = optimizer.param_groups[0]['lr']

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels, lengths in train_loader:
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = focal_loss(outputs, labels, alpha=class_weights)
        #loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    # ----- Validación -----
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for sequences, labels, lengths in val_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs = model(sequences, lengths)
            loss = focal_loss(outputs, labels, alpha=class_weights)
            #loss = criterion(outputs, labels)
            val_running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    if current_lr < prev_lr:
        print(f"Learning rate reducido a {current_lr:.6f}", flush=True)
        prev_lr = current_lr

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}", flush=True)
    
    if current_lr < 0.000010:
        print("Learning rate demasiado bajo. Terminando entrenamiento.", flush=True)
        break

    # ----- Early Stopping -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'/home/gerardo/LSE_HEALTH/LSE_TFG/models/LSTM/BiLSTM-PADDED-NORMALICED-WEIGTHED-FOCAL-LOSS-GAMMA-3_5-.pth')
        epochs_without_improvement = 0
        best_model_state = model.state_dict()
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping triggered.")
        break

print("------------- Model Evaluation ------------", flush=True)

# -------- Evaluación --------
model.load_state_dict(best_model_state)
model.eval()
test_loss = 0.0
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels, lengths in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        outputs = model(sequences, lengths)
        loss = focal_loss(outputs, labels, alpha=class_weights)
        #loss = criterion(outputs, labels)
        test_loss += loss.item() * sequences.size(0)

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_loss = test_loss / total
accuracy = correct / total

print(f"Test Loss: {avg_loss:.4f}", flush=True)
print(f"Test Accuracy: {accuracy:.4f}", flush=True)

report = classification_report(all_labels, all_preds, zero_division=0)
print("Classification Report:", flush=True)
print(report, flush=True)