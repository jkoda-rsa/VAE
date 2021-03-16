import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SyntheticReader:
    def __init__(self, train_path, test_path, label_path, error_dimension_path):
        self.X_train = pd.read_csv(train_path, header=None)
        self.X_test = pd.read_csv(test_path, header=None)
        self.labels = pd.read_csv(label_path, header=None)
        self.error_dimensions = pd.read_csv(error_dimension_path, header=None)


class TensorLoaderSynthetic:
    def __init__(self, train_path, test_path, label_path, error_dimension_path):
        self.data_train = DataLoaderHealthySynthetic(train_path)
        self.data_test = DatasetUnhealthySynthetic(test_path, label_path, error_dimension_path).data


class DataLoaderHealthySynthetic(Dataset):
    def __init__(self, train_path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all unlabelled data.
        """
        X_train = pd.read_csv(train_path, header=None)
        X_train_splitted = []

        """Yield successive n-sized chunks from lst."""
        n = 10
        for i in range(0, len(X_train), n):
            X_train_splitted.append(X_train[:][i:i + n])
        self.X_train = X_train_splitted

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return {'time_serie': torch.tensor(self.X_train[idx].astype(np.float32).values),
                'labels': torch.tensor(np.ones(len(self.X_train[idx].astype(np.float32))))}


class DatasetUnhealthySynthetic:
    def __init__(self, test_path, label_path, error_dimension_path):
        X_test = pd.read_csv(test_path, header=None)
        labels = pd.read_csv(label_path, header=None)
        error_dimensions = pd.read_csv(error_dimension_path, header=None)
        self.data = {'time_serie': torch.tensor(X_test.astype(np.float32).values), 'labels': torch.tensor(labels[0].astype(np.float32).values),
                     'error_dimensions': error_dimensions}
