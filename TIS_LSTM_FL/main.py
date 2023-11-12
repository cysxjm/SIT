import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

import flwr as fl

from dataprocess import prepare_dataset
from sampling import iid_onepass
from client import get_client_fn
from server import get_on_fit_config, get_evaluate_fn


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # 1. Prepare config and hyperparameters
    print(cfg)

    # 2. Prepare the dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_client, cfg.batch_size)
    print('Loaders ready')
    # 3. Define clients
    client_fn = get_client_fn(trainloaders, validationloaders, modelsize=323)

    # 4. define the federated strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.0001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.0001,
                                         min_available_clients=cfg.num_client,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(modelsize=323, testloader=testloader))

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_client,
        config=fl.server.ServerConfig(num_rounds=cfg.num_round),
        strategy=strategy,
        client_resources={'num_cpus': 4, 'num_gpus': 1}
    )


if __name__ == "__main__":
    main()
