import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last=True):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_last = output_last

    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)

        if self.output_last:
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State


def train_lstm(net, trainloader, optimizer, epochs, device:str):
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr)
    loss_func = nn.MSELoss()
    losses_train = []

    losses_epochs_train = []
    for epoch in range(epochs):
        losses_epoch_train = []
        total_count = 0
        for batch_idx, (data, labels) in enumerate(trainloader):
            data = data.detach().clone().type(torch.FloatTensor)
            data, labels = data.to(device), labels.to(device)
            net.zero_grad()
            outputs = net(data)
            loss_train = loss_func(outputs, torch.squeeze(labels))
            losses_train.append(loss_train.data)

            losses_epoch_train.append(loss_train.data)
            total_count += labels.size(0)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
    #
    #     avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
    #     losses_epochs_train.append(avg_losses_epoch_train.tolist())
    # return net.state_dict(), sum(losses_epochs_train) / len(losses_epochs_train)


def test_lstm(net_g, testloader, device: str):
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    losses_mse = []
    losses_mae = []
    for idx, (data, target) in enumerate(testloader):
        data = data.detach().clone().type(torch.FloatTensor)
        if device != 'gpu':
            data, target = data.cuda(), target.cuda()
        outputs = net_g(data)
        loss_MSE = torch.nn.MSELoss()
        loss_MAE = torch.nn.L1Loss()
        loss_mse = loss_MSE(outputs, torch.squeeze(target))
        loss_mae = loss_MAE(outputs, torch.squeeze(target))
        losses_mse.append(loss_mse.cpu().data.numpy())
        losses_mae.append(loss_mae.cpu().data.numpy())

    loss_mae_mean = np.mean(losses_mae)
    loss_mse_mean = np.mean(losses_mse)

    return loss_mse_mean, loss_mae_mean
