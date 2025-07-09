import pandas as pd
import numpy as np
from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler

from models.LSTM_TRIPLETS.BiLSTMTRIPLETS import LSE_Bidirectional_LSTM_Triplet
from data_preparations.windowed_csv_dataset import WindowedCSVDataset

# ----------------- Collate Function for Single Samples -----------------
def collate_single(batch):
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    lengths, sort_idx = lengths.sort(descending=True)
    sequences_padded = sequences_padded[sort_idx]
    labels = labels[sort_idx]
    return sequences_padded, labels, lengths

# ----------------- ClassBalancedSampler -----------------
class ClassBalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_to_indices = {}
        for idx in range(len(dataset)):
            label = dataset[idx][1]
            self.class_to_indices.setdefault(label, []).append(idx)
        self.labels = list(self.class_to_indices.keys())
    def __iter__(self):
        max_samples = max(len(indices) for indices in self.class_to_indices.values())
        balanced_indices = []
        for label in self.labels:
            indices = self.class_to_indices[label]
            repeated = np.random.choice(indices, max_samples, replace=True)
            balanced_indices.extend(repeated)
        np.random.shuffle(balanced_indices)
        return iter(balanced_indices)
    def __len__(self):
        return len(self.labels) * max(len(indices) for indices in self.class_to_indices.values())

# ----------------- Pairwise Distance -----------------
def pairwise_distance(embeddings):
    dot = torch.matmul(embeddings, embeddings.t())
    sq_norm = torch.diag(dot)
    dist = sq_norm.unsqueeze(1) - 2*dot + sq_norm.unsqueeze(0)
    dist = torch.clamp(dist, min=0.0)
    return torch.sqrt(dist + 1e-16)

# ----------------- Semi-Hard Triplet Loss -----------------
def batch_semi_hard_triplet_loss(labels, embeddings, margin, device):
    pdist = pairwise_distance(embeddings)
    labels = labels.unsqueeze(1)
    mask_pos = (labels == labels.t()).float() - torch.eye(labels.size(0), device=device)
    mask_neg = (labels != labels.t()).float()
    losses = []
    N = embeddings.size(0)
    for i in range(N):
        pos_idx = (mask_pos[i] > 0).nonzero(as_tuple=False).view(-1)
        if pos_idx.numel() == 0:
            continue
        for j in pos_idx:
            dij = pdist[i, j]
            valid_neg = pdist[i][(mask_neg[i] > 0) & (pdist[i] > dij) & (pdist[i] < dij + margin)]
            if valid_neg.numel() > 0:
                d_an = torch.min(valid_neg)
                losses.append(dij - d_an + margin)
    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=device)
    return torch.stack(losses).mean()

# ----------------- Batch-Hard Triplet Loss -----------------
def batch_hard_triplet_loss(labels, embeddings, margin, device):
    pdist = pairwise_distance(embeddings)
    labels = labels.unsqueeze(1)
    mask_pos = (labels == labels.t()).float() - torch.eye(labels.size(0), device=device)
    mask_neg = (labels != labels.t()).float()
    # hardest positive per anchor
    hardest_pos = (pdist * mask_pos).max(dim=1).values
    # hardest negative: mask non-negatives with large value
    max_val = pdist.max().detach()
    masked_neg = pdist + max_val * (1.0 - mask_neg)
    hardest_neg = masked_neg.min(dim=1).values
    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()

# ----------------- Load CSV file list -----------------
def load_file_list(path):
    return pd.read_csv(path).iloc[:,0].dropna().tolist()

# ----------------- Main Pipeline -----------------
if __name__ == '__main__':
    # Paths
    base_dir = '/home/gerardo/FEATURES_POSE_DATASET'
    split_dir = '/home/gerardo/LSE_HEALTH/LSE_TFG/train_test_val_split'
    train_files = load_file_list(f'{split_dir}/train_weigthed_samples.csv')
    val_files   = load_file_list(f'{split_dir}/val_weigthed_samples.csv')
    test_files  = load_file_list(f'{split_dir}/test_weigthed_samples.csv')

    # Dataset args
    common = {'drop_columns':['frame'], 'file_list': train_files}
    extra = {'target_frames':11,'stride':5}

    train_ds = WindowedCSVDataset(root_dir=base_dir, **common, **extra)
    common['file_list'] = val_files
    val_ds   = WindowedCSVDataset(root_dir=base_dir, **common, **extra)
    common['file_list'] = test_files
    test_ds  = WindowedCSVDataset(root_dir=base_dir, **common, **extra)

    # Hyperparams
    seq, _, _ = train_ds[0]
    input_size = seq.shape[1]
    hidden_size = input_size // 3
    num_layers=2; dropout=0.5; bidir=True; embedding_dim=128

    # Dataloaders
    batch_size=16
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=ClassBalancedSampler(train_ds), collate_fn=collate_single)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_single)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_single)

    # Model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSE_Bidirectional_LSTM_Triplet(input_size, hidden_size, num_layers,
                                          embedding_dim=embedding_dim, dropout=dropout, bidirectional=bidir)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)
    margin=0.5
    early_stop=15; patience=0; best_val=float('inf'); min_lr=1e-5

    # Training loop
    for epoch in range(1, 151):
        # Train
        model.train()
        tloss=0
        for seqs, labels, lengths in train_loader:
            seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
            opt.zero_grad()
            emb = model(seqs, lengths)
            loss = batch_semi_hard_triplet_loss(labels, emb, margin, device)
            loss.backward(); opt.step()
            tloss += loss.item()*seqs.size(0)
        tloss /= len(train_loader.dataset)
        # Val
        model.eval(); vloss=0
        with torch.no_grad():
            for seqs, labels, lengths in val_loader:
                seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
                emb = model(seqs, lengths)
                loss = batch_semi_hard_triplet_loss(labels, emb, margin, device)
                vloss += loss.item()*seqs.size(0)
        vloss /= len(val_loader.dataset)
        sched.step(vloss)
        lr=opt.param_groups[0]['lr']
        print(f"Epoch {epoch} | Train: {tloss:.4f} | Val: {vloss:.4f} | LR: {lr:.6f}")
        # Early stop
        if vloss < best_val:
            best_val=vloss; best_state=model.state_dict(); patience=0
        else:
            patience+=1
        if patience>=early_stop or lr<min_lr:
            print("Stopping..."); break

    # Load best
    model.load_state_dict(best_state)
    torch.save(best_state, '/home/gerardo/LSE_HEALTH/LSE_TFG/models/LSTM_TRIPLETS/Bi-LSTM-TRIPLET-SEMI-HARD-.pth')

    # Embeddings function
    def extract(ldr):
        model.eval(); embs=[]; labs=[]
        with torch.no_grad():
            for seqs, labels, lengths in ldr:
                seqs, lengths = seqs.to(device), lengths.to(device)
                emb = model(seqs, lengths).cpu()
                embs.append(emb); labs.append(labels)
        return torch.cat(embs).numpy(), torch.cat(labs).numpy()

    # Eval SVM
    te, tl = extract(test_loader)
    tr, trl = extract(DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_single))
    tr = normalize(tr, norm='l2'); te = normalize(te, norm='l2')
    params={'C':[0.1,1,10,100],'gamma':['scale',0.01,0.001,0.0001]}
    gs=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'), params, cv=5, n_jobs=-1)
    gs.fit(tr, trl)
    print("Best SVM:", gs.best_params_, gs.best_score_)
    pred=gs.best_estimator_.predict(te)
    print(classification_report(tl, pred))
