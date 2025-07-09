import os
import torch
import pandas as pd

class CSVSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, drop_columns=None, file_list=None):
        """
        Dataset que carga archivos CSV de secuencias.

        Args:
            root_dir (str): Carpeta raíz con la estructura class_name/archivo.csv
            drop_columns (list): Columnas a eliminar como ['frame']
            file_list (list, optional): Lista de rutas relativas 'clase/archivo.csv' que se deben incluir
        """
        self.root_dir = root_dir
        self.drop_columns = drop_columns or []
        self.samples = []
        self.class_to_idx = {}

        file_set = set(file_list) if file_list else None

        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            self.class_to_idx[class_name] = class_idx

            for filename in os.listdir(class_dir):
                if not filename.endswith('.csv'):
                    continue

                rel_path = f"{class_name}/{filename}"

                if file_set and rel_path not in file_set:
                    continue

                self.samples.append((rel_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_rel, label = self.samples[idx]
        file_path = os.path.join(self.root_dir, file_rel)

        df = pd.read_csv(file_path)

        for col in self.drop_columns:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        sequence = torch.tensor(df.values, dtype=torch.float32)
        length = sequence.size(0)

        return sequence, label, length
