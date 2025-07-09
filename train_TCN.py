import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from models.TCN.ed_tcn import ED_TCN

# ---------------------------
# Configuración general
# ---------------------------
BASE_FOLDER      = '/home/gerardo/LSE_SEGMENTATION_SPLIT'
checkpoint_path  = '/home/gerardo/LSE_HEALTH/LSE_TFG/models/TCN/binary_detector_.pth'

batch_size       = 8
num_epochs       = 100
learning_rate    = 0.001
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ignore_index     = -100

# ---------------------------
# Dataset y función de collate
# ---------------------------
class GestureDataset(Dataset):
    def __init__(self, csv_folder: str, label_col: str = 'Gesture'):
        self.files = [
            os.path.join(csv_folder, f)
            for f in os.listdir(csv_folder)
            if f.lower().endswith('.csv')
        ]
        self.label_col = label_col
        df0 = pd.read_csv(self.files[0])
        self.feature_cols = [c for c in df0.columns if c != label_col]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_csv(self.files[idx])
        x = torch.from_numpy(df[self.feature_cols].values).float()
        y = torch.from_numpy(df[self.label_col].values).long()
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=ignore_index)
    return x_padded, y_padded, None

class GestureBinaryDataset(GestureDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        y_bin = (y != 43).long()
        return x, y_bin

# ---------------------------
# Focal Loss con pesos de clase
# ---------------------------
def focal_loss(inputs, targets, alpha=None, gamma=2):
    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)
    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
    pt = torch.sum(prob * targets_one_hot, dim=-1)
    at = alpha[targets] if alpha is not None else 1.0
    loss = -at * (1 - pt) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)
    return loss.mean()

def get_class_weights(dataset, device):
    labels = []
    for _, y in dataset:
        labels.extend(y.tolist())
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [ total/(2 * counts[i]) for i in (0,1) ]
    return torch.tensor(weights, dtype=torch.float, device=device)

# ---------------------------
# Entrenamiento y evaluación
# ---------------------------
def train_model(model, train_loader, val_loader, class_weights, optimizer, scheduler, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    best_val = float('inf')
    epochs_no_improve = 0
    patience = 5

    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch) 
            B, T, C = logits.shape
            y_trunc = y_batch[:, :T]

            logits_flat  = logits.reshape(-1, C)
            targets_flat = y_trunc.reshape(-1)
            mask = targets_flat != ignore_index

            loss = focal_loss(
                logits_flat[mask],
                targets_flat[mask],
                alpha=class_weights
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, _ in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                B, T, C = logits.shape
                y_trunc = y_batch[:, :T]

                logits_flat  = logits.reshape(-1, C)
                targets_flat = y_trunc.reshape(-1)
                mask = targets_flat != ignore_index

                val_loss += focal_loss(
                    logits_flat[mask],
                    targets_flat[mask],
                    alpha=class_weights
                ).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping después de {epoch} épocas sin mejora.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model

def evaluate_model(model, test_loader, class_weights):
    model.eval()
    test_loss = 0.0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x_batch, y_batch, _ in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            B, T, C = logits.shape
            y_trunc = y_batch[:, :T]

            logits_flat  = logits.reshape(-1, C)
            targets_flat = y_trunc.reshape(-1)
            mask = targets_flat != ignore_index

            test_loss += focal_loss(
                logits_flat[mask],
                targets_flat[mask],
                alpha=class_weights
            ).item()

            preds = logits.argmax(-1).reshape(-1).cpu()
            trues = targets_flat.cpu()
            all_preds.extend(preds[mask.cpu()].tolist())
            all_trues.extend(trues[mask.cpu()].tolist())

    test_loss /= len(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}\n")
    print(classification_report(all_trues, all_preds,
                                labels=[0,1],
                                target_names=["Fondo","Gesto"],
                                zero_division=0))

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train_ds = GestureBinaryDataset(os.path.join(BASE_FOLDER, 'train'))
    val_ds   = GestureBinaryDataset(os.path.join(BASE_FOLDER, 'val'))
    test_ds  = GestureBinaryDataset(os.path.join(BASE_FOLDER, 'test'))

    class_weights = get_class_weights(train_ds, device)

    sample_weights = []
    for i in range(len(train_ds)):
        _, y = train_ds[i]
        sample_weights.append(class_weights[y].float().mean().item())
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=sampler,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    model = ED_TCN(n_nodes=[64, 128],
                   conv_len=5,
                   n_classes=2,
                   n_feat=len(train_ds.feature_cols)
                  ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=4,
                                  verbose=True)

    print("=== Entrenando detector binario ===")
    model = train_model(model,
                        train_loader,
                        val_loader,
                        class_weights,
                        optimizer,
                        scheduler,
                        checkpoint_path)

    print("=== Evaluación final detector binario ===")
    evaluate_model(model, test_loader, class_weights)
