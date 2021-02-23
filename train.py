import torch

import torch.nn as nn
import argparse
from rossler_map import RosslerMap
import numpy as np
from tqdm import tqdm

model = nn.Sequential(nn.Linear(3, 10),
                           # nn.ReLU(),
                           nn.Linear(10, 5),
                           # nn.ReLU(),
                           nn.Linear(5, 3))


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default='basic')
    parser.add_argument("--batch_size", type=float, default=30)
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=20)


    args = parser.parse_args()

    # model = nn.Sequential(nn.Linear(3, 10),
    #                        nn.ReLU(),
    #                        nn.Linear(10, 5),
    #                        nn.ReLU(),
    #                        nn.Linear(5, 3))

    critirion = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)


    train(model=model, critirion=critirion, epochs=args.epochs,
          optimizer=optimizer, exp_name=args.exp_name, batch_size=args.batch_size,
          dataset_size=args.dataset_size, log_interval=args.log_interval)

