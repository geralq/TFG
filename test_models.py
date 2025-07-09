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
optimizer = optim.Adam(model.parameters(), lr=0.001)
# -------- Calcular pesos de clase --------
class_weights = get_class_weights(train_dataset, device)
print("Class Weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)

model.load_state_dict(torch.load('/home/gerardo/LSE_HEALTH/LSE_TFG/models/LSTM/BiLSTM-PADDED-NORMALICED-WEIGTHED-FOCAL-LOSS-GAMMA-3_5-77.pth'))

model.to(device)

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
        # loss = criterion(outputs, labels)
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

report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
report_text = classification_report(all_labels, all_preds, zero_division=0)

# Calcular UAR: promedio de los recalls individuales de cada clase
class_recalls = [
    metrics['recall']
    for label, metrics in report_dict.items()
    if label.isdigit()
]
uar = sum(class_recalls) / len(class_recalls)

# Imprimir resultados
print("==== Métricas de Evaluación ====")

print(report_text)
print(f"Accuracy global: {accuracy:.4f}")
print(f"Test Loss: {avg_loss:.4f}")
print(f"UAR (Unweighted Average Recall): {uar:.4f}")