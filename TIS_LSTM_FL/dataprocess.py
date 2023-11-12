import numpy as np
import torch
from ray.data import Dataset

from LOOPDataset import LOOPDataset
from torch.utils.data import DataLoader, random_split

from sampling import iid_onepass


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    # def __getitem__(self, item):
    #     image, label = self.dataset[self.idxs[item]]
    #     return image, label

    def iter_batches(self, batch_size, shuffle=True, num_workers=0):
        if shuffle:
            np.random.shuffle(self.idxs)

        for i in range(0, len(self.idxs), batch_size):
            batch_idxs = self.idxs[i:i + batch_size]
            yield [(self.dataset[j][0], self.dataset[j][1]) for j in batch_idxs]

    def iter_rows(self):
        for idx in self.idxs:
            yield self.dataset[idx][0], self.dataset[idx][1]


def get_loop(data_path="../data/loop/"):

    trainset = LOOPDataset(data_path, phase='train')
    testset = LOOPDataset(data_path, phase='eval')

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):

    trainset, testset = get_loop()

    # split trainset into iid

    # num_data = len(trainset) // num_partitions
    #
    # partition_len = [num_data] * num_partitions
    #
    # trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    dict_clients, dict_test = iid_onepass(trainset, 500, testset, 125, num_partitions, dataset_name='loop')
    trainsets = DatasetSplit(trainset, dict_clients)
    # testset = DatasetSplit(testset, dict_test)

    # create dataloaders
    trainloaders = []
    valloaders = []
    # for trainset_ in trainsets:
    #     num_total = len(trainset_)
    #     num_val = int(val_ratio*num_total)
    #     num_train = num_total - num_val
    #
    #     for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))
    #
    #     trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
    #     valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    for batch in trainsets.iter_batches(batch_size, shuffle=True, num_workers=2):
        num_total = len(batch)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(batch, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(testset, batch_size=2*batch_size)

    return trainloaders, valloaders, testloader