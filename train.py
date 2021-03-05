import torch

import torch.nn as nn
import argparse
from rossler_map import RosslerMap
import numpy as np
from tqdm import tqdm

model = nn.Sequential(nn.Linear(3, 50),
                           # nn.ReLU(),
                           nn.Linear(50, 10),
                           # nn.ReLU(),
                           nn.Linear(10, 5),
                           # nn.ReLU(),
                           nn.Linear(5, 3))
mse_loss = nn.MSELoss()


def train(model=None, critirion=None, optimizer=None, epochs=10, dataset_size=1000,
          batch_size=10, log_interval=20, exp_name='test'):

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
            loss = critirion(preds, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % log_interval == 0:

            print('Loss : ', loss.item())
            model_file = exp_name + 'model_'+ str(epoch) +'.pth'
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



    args = parser.parse_args()

    # model = nn.Sequential(nn.Linear(3, 10),
    #                        nn.ReLU(),
    #                        nn.Linear(10, 5),
    #                        nn.ReLU(),
    #                        nn.Linear(5, 3))



    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)


    train(model=model, critirion=loss_v0, epochs=args.epochs,
          optimizer=optimizer, exp_name=args.exp_name, batch_size=args.batch_size,
          dataset_size=args.dataset_size, log_interval=args.log_interval)

