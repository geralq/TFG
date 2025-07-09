import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class WindowedCSVDataset(Dataset):
    def __init__(self, root_dir, drop_columns=None, target_frames=22, stride=11, pad_value=0, file_list=None):
        """
        Dataset que genera ventanas desde archivos CSV:
        - Si 'file_list' está definido, solo incluirá esos archivos (rutas relativas).
        - Si una secuencia es menor a target_frames, se le aplica padding.
        - Devuelve (tensor, class_idx, longitud real sin padding).
        """
        self.root_dir = root_dir
        self.drop_columns = drop_columns or []
        self.target_frames = target_frames
        self.stride = stride
        self.pad_value = pad_value
        self.samples = []  # (filepath, class_idx, start_index)

        self.file_list = set(file_list) if file_list else None

        self.class_to_idx = {}
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for class_idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = class_idx
            class_dir = os.path.join(root_dir, class_name)

            for filename in os.listdir(class_dir):
                if not filename.endswith(".csv"):
                    continue

                relative_path = f"{class_name}/{filename}"
                if self.file_list and relative_path not in self.file_list:
                    continue 

                filepath = os.path.join(class_dir, filename)
                df = pd.read_csv(filepath)

                for col in self.drop_columns:
                    if col in df.columns:
                        df.drop(columns=col, inplace=True)

                n_frames = len(df)

                if n_frames >= self.target_frames:
                    for start in range(0, n_frames - self.target_frames + 1, self.stride):
                        self.samples.append((filepath, class_idx, start))
                    if (n_frames - self.target_frames) % self.stride != 0:
                        self.samples.append((filepath, class_idx, n_frames - self.target_frames))
                else:
                    self.samples.append((filepath, class_idx, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, class_idx, start = self.samples[idx]
        df = pd.read_csv(filepath)

        for col in self.drop_columns:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        n_frames = len(df)

        if n_frames >= self.target_frames:
            window = df.iloc[start:start + self.target_frames].values
            length = self.target_frames
        else:
            pad_len = self.target_frames - n_frames
            padding = [[self.pad_value] * df.shape[1]] * pad_len
            window = df.values.tolist() + padding
            length = n_frames

        sequence_tensor = torch.tensor(window, dtype=torch.float32)
        return sequence_tensor, class_idx, length
