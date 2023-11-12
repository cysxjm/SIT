import flwr as fl

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from flwr.common import NDArrays, Scalar
from network import LSTM, test_lstm
from torch.utils.data import DataLoader
from network import train_lstm


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, modelsize) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = LSTM(modelsize, modelsize, modelsize, output_last=True)
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, parameters):
        """With the model paramters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and conver them to a list of
        NumPy arryas. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # read from config
        lr, epochs = config["lr"], config["epochs"]

        # Define the optimizer
        # optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)

        # do local training
        train_lstm(self.model, self.trainloader, optimizer, epochs, self.device)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        # Evaluate
        loss_mse_mean, loss_mae_mean = test_lstm(self.model, self.valloader, device=self.device)

        # Return statistics
        return float(loss_mse_mean), float(loss_mae_mean),


def get_client_fn(trainloader, valloaders, modelsize):
    """Return a function to construct a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Create and return client
        return FlowerClient(trainloader=trainloader[int(cid)],
                            valloader=valloaders[int(cid)],
                            modelsize=modelsize)

    return client_fn


