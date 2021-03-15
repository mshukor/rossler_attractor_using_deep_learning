import torch
import torch.nn as nn
import numpy as np

class My_Model(nn.Module):
    def __init__(self, nf=100):
        super(My_Model, self).__init__()
        
        self.n_features = nf
        self.model = nn.Sequential( nn.Linear(3, nf),      nn.ReLU(),
                                    nn.Linear(nf, nf*2),   nn.ReLU(),
                                    nn.Linear(nf*2, nf*3), nn.ReLU(),
                                    nn.Linear(nf*3, 3)
                                  )

    def forward(self, x, jacob=False):
        if jacob:
            x = torch.tensor(x).view(1,-1)
            x = self.model(x)
            return x.detach().numpy()
        return self.model(x)
        
    def batch_model_jacobian(self, batch, create_graph=True):
        
        in_traj = torch.tensor(batch).view(1,-1)
        print(in_traj)
        in_traj.requires_grad = True
        out = self.model(in_traj)

        outputs = tuple(out.sum(dim=0))
        j = torch.zeros((in_traj.shape[0], 3, 3))
        for i in range(3):
            j[:, i, :] = torch.autograd.grad( outputs[i], in_traj, create_graph=create_graph, 
                                              only_inputs=True, allow_unused=True)[0]
        
        return j.detach().numpy()


class GRUNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=500, output_dim=1, 
                       n_layers=2, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU( input_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden
