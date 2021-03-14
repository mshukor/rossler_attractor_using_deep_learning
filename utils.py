import torch

from torch.utils.data import Dataset, DataLoader, random_split
from rossler_map import RosslerMap


class RDataset(Dataset):
    def __init__(self, size, init, delta_t, history, fake_prop):
        super().__init__()
        self.size = size
        self.init = init
        self.delta_t = delta_t
        self.history = history
        self.mask = torch.ones((self.history + 2, 1))
        self.mask[-1, 0] = 0  # Set target to 0

        self.rm = RosslerMap(delta_t=delta_t)

        self.w, self.t = self.rm.full_traj(size, init)
        self.w = torch.tensor(self.w).float()

    def __len__(self):
        return self.nb_samples - 1 - self.history

    def __getitem__(self, idx):
        return self.w[idx : idx + self.history + 2]