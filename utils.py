import torch
import torch.nn as nn
from torch.utils.data import Dataset

from rossler_map import RosslerMap


class RDataset(Dataset):
    def __init__(self, size, init, delta_t, history, scale=False):
        super().__init__()
        self.size = size
        self.init = init
        self.delta_t = delta_t
        self.history = history
        self.mask = torch.ones((self.history + 2, 1))
        self.mask[-1, 0] = 0  # Set target to 0

        self.rm = RosslerMap(delta_t=delta_t)

        self.w, self.t = self.rm.full_traj(size, init)
        if scale:
          scaler =MinMaxScaler(feature_range=(-1, 1))
          self.w = scaler.fit_transform(self.w)

        self.w = torch.tensor(self.w).float()

    def __len__(self):
        return self.size - 1 - self.history

    def __getitem__(self, idx):
        return self.w[idx : idx + self.history + 2]
        

# Loss functions
def loss_v0(preds, gt, criterion=nn.MSELoss()):
    return criterion(preds, gt)

def loss_v1(preds, gt, criterion=nn.MSELoss()):

    d_preds = (preds[1:] - preds[:-1]) / args.dt
    d_gt = (gt[1:] - gt[:-1]) / args.dt

    loss = criterion(preds, gt) + args.lamda*criterion(d_preds, d_gt)
    return loss

def loss_v2(preds, gt,  criterion=nn.MSELoss()):

    d_preds = (preds[1:] - preds[:-1]) / args.dt

    loss = criterion(preds, gt) + args.lamda*torch.sqrt(torch.sum(d_preds**2))
    return loss
