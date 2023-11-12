from datetime import datetime

from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from spliterator import spliterator
from test import test_lstm
from network import LSTM
from option import args_parser



if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    writer = SummaryWriter()

    if args.dataset == 'loop':
        data_spliter = spliterator(data_path="../data/loop/")
        dataset_train, dataset_test = data_spliter.get_sensordata()

    else:
        exit('Error: unrecognized dataset')

    net_global = {}
    for key in dataset_train:
        print('The ID of the highway now is ', key)
        writer = SummaryWriter(log_dir=f'logs/network{key}')
        image_size = dataset_train[key][0][0].shape
        net_global[key] = LSTM(image_size[1], image_size[1], image_size[1], output_last=True).to(args.device)
        print(net_global[key])
        optimizer = torch.optim.RMSprop(net_global[key].parameters(), lr=args.lr)
        loss_func = nn.MSELoss()
        losses_train = []
        ldr_train = DataLoader(dataset_train[key], batch_size=args.local_bs, shuffle=True)

        start_time = datetime.now()
        for epoch in range(args.local_ep):
            losses_epoch_train = []
            minutes_since_start = (datetime.now() - start_time).total_seconds() / 60
            for batch_idx, (data, labels) in enumerate(ldr_train):
                data = data.detach().clone().type(torch.FloatTensor)
                data, labels = data.to(args.device), labels.to(args.device)
                net_global[key].zero_grad()
                outputs = net_global[key](data)
                loss_train = loss_func(outputs, torch.squeeze(labels))
                losses_train.append(loss_train.data)
                losses_epoch_train.append(loss_train.data)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
            avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
            print('Round {:3d}, Average loss {:.3f} '.format(epoch, avg_losses_epoch_train))
            writer.add_scalar(f'Loss/epoch_{key}', avg_losses_epoch_train, epoch)
            writer.add_scalar(f'Loss/time_{key}', avg_losses_epoch_train, global_step=minutes_since_start)
        writer.close()

        net_global[key].eval()
        train_loss_mse, train_loss_mae = test_lstm(net_global[key], dataset_train[key], args)
        test_loss_mse, test_loss_mae = test_lstm(net_global[key], dataset_test[key], args)
        print("Training MSE Loss is: {:.2f}", train_loss_mse, "Training MAE Loss: {:.2f}", train_loss_mae)
        print("Testing MSE Loss: {:.2f}", test_loss_mse, "Testing MAE Loss: {:.2f}", test_loss_mae)
