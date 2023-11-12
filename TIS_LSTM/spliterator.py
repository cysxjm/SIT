import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from LOOPDataset import LOOPDataset


class spliterator(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data, self.targets = self.get_sensordata()

    def get_sensordata(self):
        speed_matrix = pd.read_pickle(os.path.join(self.data_path, 'speed_matrix_2015'))
        time_len = speed_matrix.shape[0]

        # Normalization
        max_speed = speed_matrix.max().max()
        speed_matrix = speed_matrix / max_speed

        highway_codes = ['005', '090', '405', '520']
        speed_matrix_highway = {}
        dataset_train = {}
        dateset_test = {}
        for code in highway_codes:
            speed_matrix_highway[code] = speed_matrix.loc[:, speed_matrix.columns.str[1:4] == code]
            dataset_train[code] = LOOPDataset(speed_matrix_highway[code], phase='train')
            dateset_test[code] = LOOPDataset(speed_matrix_highway[code], phase='eval')
        return dataset_train, dateset_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]