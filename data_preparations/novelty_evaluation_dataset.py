import pandas as pd
import torch
from torch.utils.data import Dataset

class NoveltyEvaluationDataset(Dataset):
    def __init__(self, csv_path, drop_columns=None, window_size=22, stride=10):
        self.window_size = window_size
        self.stride = stride

        df = pd.read_csv(csv_path)
        
        # Extract frame values if available (before dropping)
        if 'frame' in df.columns:
            frame_values = df['frame'].values
        else:
            frame_values = None

        if drop_columns:
            df = df.drop(columns=[col for col in drop_columns if col in df.columns])

        self.sequences = []
        self.frame_ranges = []
        n_frames = len(df)

        for start in range(0, n_frames - window_size + 1, stride):
            end = start + window_size
            window_df = df.iloc[start:end]
            tensor = torch.tensor(window_df.values, dtype=torch.float32)
            self.sequences.append(tensor)
            
            if frame_values is not None:
                self.frame_ranges.append((int(frame_values[start]), int(frame_values[end - 1])))
            else:
                self.frame_ranges.append((None, None))
                
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        length = sequence.size(0)
        frame_range = self.frame_ranges[idx]
        return sequence, length, frame_range