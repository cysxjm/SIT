import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class LOOPDataset(Dataset):
    def __init__(self, speed_matrix, phase="train"):
        self.speed_matrix = speed_matrix
        self.phase = phase
        self.data, self.targets = self.get_data()

    # def extract_highway_code(id):
    #     return id[1:4]

    def get_data(self, seq_len=10, pred_len=1, train_proportion=0.7):
        # speed_matrix = pd.read_pickle(os.path.join(self.data_path, 'speed_matrix_2015'))

        time_len = self.speed_matrix.shape[0]

        # Normalization
        max_speed = self.speed_matrix.max().max()
        speed_matrix = self.speed_matrix / max_speed
        # filtered_data_520, filtered_data_005, filtered_data_090, filtered_data_405 = [
        #     speed_matrix.loc[:, speed_matrix.columns.str[1:4] == code] for code in ['520', '005', '090', '405']]
        #
        # speed_matrix_highway_code = [filtered_data_520, filtered_data_005, filtered_data_090, filtered_data_405]

        speed_sequences, speed_labels = [], []
        for i in range(time_len - seq_len - pred_len):
            speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
            speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
        speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)

        # shuffle and split the dataset to training and testing datasets
        sample_size = speed_sequences.shape[0]
        index = np.arange(sample_size, dtype=int)
        np.random.shuffle(index)

        train_index = int(np.floor(sample_size * train_proportion))

        if self.phase == "train":
            data, label = speed_sequences[:train_index], speed_labels[:train_index]
            data = torch.Tensor(data).detach().clone().type(torch.FloatTensor)
            label = torch.Tensor(label).detach().clone().type(torch.FloatTensor)
        else:
            data, label = speed_sequences[train_index:], speed_labels[train_index:]
            data = torch.Tensor(data).detach().clone().type(torch.FloatTensor)
            label = torch.Tensor(label).detach().clone().type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# if __name__ == '__main__':
#     real_path = os.path.dirname(os.path.realpath(__file__))
#     loop_data_path = os.path.join(real_path, "../../data/loop/")
#     dataset = LOOPDataset(data_path=loop_data_path)
#     print(len(dataset))
