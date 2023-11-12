from collections import OrderedDict

import torch

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from network import LSTM, test_lstm


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {'lr': config.lr, 'epochs': config.epochs}
    return fit_config_fn


def get_evaluate_fn(modelsize: int, testloader):

    def evaluate_fn(server_round: int, parameters, config):
        model = LSTM(modelsize, modelsize, modelsize, output_last=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss_mse_mean, loss_mae_mean = test_lstm(model, testloader, device)

        return {"loss_mse_mean": loss_mse_mean}, {"loss_mae_mean": loss_mae_mean}

    return evaluate_fn