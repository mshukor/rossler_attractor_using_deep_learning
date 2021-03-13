import torch

import torch.nn as nn
import argparse
from rossler_map import RosslerMap
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

model = nn.Sequential(nn.Linear(3, 50),
                           nn.ReLU(),
                           nn.Linear(50, 10),
                           nn.ReLU(),
                           nn.Linear(10, 5),
                           nn.ReLU(),
                           nn.Linear(5, 3))


class GRUNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=500, output_dim=1, n_layers=2, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()#.to(device)
        return hidden


rnn_hidden = 500
rnn_num_layers = 2
rnn_history = 32
model_rnn = GRUNet(input_dim=1, hidden_dim=rnn_hidden, output_dim=1, n_layers=rnn_num_layers)

mse_loss = nn.MSELoss()

def train_rnn(model=None, criterion=None, optimizer=None, epochs=10, dataset_size=1000,
          batch_size=10, log_interval=20, exp_name='test', device=None):

    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)

    model.to(device)
    model.train()
    scaler =MinMaxScaler(feature_range=(-1, 1))

    for epoch in tqdm(range(epochs)):

        # INIT = np.array([-5.75, -1.6, 0.02])
        INIT = np.random.rand(3)*20 -10
        traj, t = ROSSLER_MAP.full_traj(dataset_size, INIT)

        nb_steps = int(len(traj) / rnn_history)

        traj = scaler.fit_transform(traj)

        traj_tensor = torch.from_numpy(traj).float()
        h = model.init_hidden(1)

        for i in range(nb_steps -1):

            batch = traj_tensor[i:i+rnn_history, 1] # (history, 1)
            batch = batch.view((1, batch.shape[0], 1))
            ## labels are just the same batch but shifted
            gt = traj_tensor[i+rnn_history, 1] # (1)

            h = h.data

            model.zero_grad()
            # optimizer.zero_grad()

            out, h = model(batch.to(device), h.to(device))

            loss = criterion(out[0][0], gt.to(device))

            loss.backward(retain_graph=False)
            optimizer.step()


        if epoch % log_interval == 0:

            print('Loss : ', loss.item())
            model_file = "models/" + exp_name + '_model_'+ str(epoch) +'.pth'
            print('saving :', model_file)
            torch.save(model.state_dict(), model_file)


def train(model=None, criterion=None, optimizer=None, epochs=10, dataset_size=1000,
          batch_size=10, log_interval=20, exp_name='test', device=None):

    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)

    for epoch in tqdm(range(epochs)):

        # INIT = np.array([-5.75, -1.6, 0.02])
        INIT = np.random.rand(3)*20 -10
        traj, t = ROSSLER_MAP.full_traj(dataset_size, INIT)

        nb_steps = int(len(traj) / batch_size)

        traj_tensor = torch.from_numpy(traj).float()

        for i in range(nb_steps -1):

            batch = traj_tensor[i:i+batch_size]

            ## labels are just the same batch but shifted
            gt = traj_tensor[i+1:i+1+batch_size]

            preds = model(batch)
            loss = criterion(preds, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % log_interval == 0:

            print('Loss : ', loss.item())
            model_file = "models/" + exp_name + '_model_'+ str(epoch) +'.pth'
            print('saving :', model_file)
            torch.save(model.state_dict(), model_file)


def loss_v0(preds, gt):
    return mse_loss(preds, gt)

def loss_v1(preds, gt):

    d_preds = (preds[1:] - preds[:-1]) / args.dt
    d_gt = (gt[1:] - gt[:-1]) / args.dt

    loss = mse_loss(preds, gt) + args.lamda*mse_loss(d_preds, d_gt)
    return loss

def loss_v2(preds, gt):

    d_preds = (preds[1:] - preds[:-1]) / args.dt

    loss = mse_loss(preds, gt) + args.lamda*torch.sqrt(torch.sum(d_preds**2))
    return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default='basic_lossv1')
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--lamda", type=float, default=1)
    parser.add_argument("--rnn", type=bool, default=True)


    args = parser.parse_args()

    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.rnn:
        optimizer = torch.optim.Adam(params=model_rnn.parameters(), lr=args.lr)

        train_rnn(model=model_rnn, criterion=loss_v0, epochs=args.epochs,
              optimizer=optimizer, exp_name=args.exp_name, batch_size=args.batch_size,
              dataset_size=args.dataset_size, log_interval=args.log_interval, device=device)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        train(model=model, criterion=loss_v0, epochs=args.epochs,
              optimizer=optimizer, exp_name=args.exp_name, batch_size=args.batch_size,
              dataset_size=args.dataset_size, log_interval=args.log_interval, device=device)

