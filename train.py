import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from models import My_Model, GRUNet
from rossler_map import RosslerMap
from utils import RDataset, loss_v0, loss_v1, loss_v2

#-------------------------------------------------------------------------------
#                               First Approach Training
#-------------------------------------------------------------------------------
def train(model=None, criterion=None, optimizer=None, epochs=10, dataset_size=1000,
          batch_size=10, log_interval=20, exp_name='test', device=None, 
          delta_t=1e-2, history=0):
    
    model.to(device)
    model.train()
    
    init = np.array([-5.75, -1.6, 0.02])
    dataset = RDataset(dataset_size, init, delta_t, history)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=False)

    for epoch in tqdm(range(epochs)):
        for (i,batch) in enumerate(train_loader):
               
            assert batch.shape[1]==2
            in_traj = batch[:,0,:].to(device)
            gt = batch[:,-1,:].to(device)  # labels are just the same batch but shifted

            preds = model(in_traj)
            loss = criterion(preds, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % log_interval == 0:
            print('Loss : ', loss.item())
            model_file = "models/" + exp_name + '_model_'+ str(epoch) +'.pth'
            print('saving :', model_file)
            torch.save(model.state_dict(), model_file)



#-------------------------------------------------------------------------------
#                              Second Approach Training
#-------------------------------------------------------------------------------
def train_rnn(model=None, criterion=None, optimizer=None, epochs=10, dataset_size=1000,
          batch_size=10, log_interval=20, exp_name='test', device=None, history=0):

    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)

    model.to(device)
    model.train()

    init = np.array([-5.75, -1.6, 0.02])
    dataset = RDataset(dataset_size, init, delta_t, history, scale=False)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=False)

    for epoch in tqdm(range(epochs)):
        h = model.init_hidden(1)
        
        for (i,batch) in enumerate(train_loader):
            
            in_traj = batch[:,:history,:].to(device)
            in_traj = in_traj.view((1, batch.shape[0], 1))
            gt = batch[:,-1,:].to(device)  # labels are just the same batch but shifted

            h = h.data

            model.zero_grad()
            # optimizer.zero_grad()

            out, h = model(in_traj.to(device), h.to(device))
            loss = criterion(out[0][0], gt.to(device))

            loss.backward(retain_graph=False)
            optimizer.step()

        if epoch % log_interval == 0:
            print('Loss : ', loss.item())
            model_file = "models/" + exp_name + '_model_'+ str(epoch) +'.pth'
            print('saving :', model_file)
            torch.save(model.state_dict(), model_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default='basic_lossv0')
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--dataset_size", type=int, default=5000000)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--history", type=float, default=0)
    parser.add_argument("--lamda", type=float, default=1)
    parser.add_argument("--rnn", type=bool, default=False)

    args = parser.parse_args()

    # If we have a GPU available, we'll set our device to GPU. 
    is_cuda = torch.cuda.is_available()
    
    if is_cuda: print("on GPU") 
    else: print("on CPU")
    
    device = torch.device("cuda") if is_cuda else torch.device("cpu")
    
    if not args.rnn:
        model = My_Model()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        
        train( model=model, criterion=loss_v0, epochs=args.epochs,
               optimizer=optimizer, exp_name=args.exp_name, batch_size=args.batch_size,
               dataset_size=args.dataset_size, log_interval=args.log_interval, 
               device=device, delta_t=args.dt, history=args.history)
    else:

        model_rnn = GRUNet( input_dim=1, hidden_dim=500, output_dim=1, n_layers=2)
        optimizer = torch.optim.Adam(params=model_rnn.parameters(), lr=args.lr)
        
        train_rnn( model=model_rnn, criterion=loss_v0, epochs=args.epochs,
                   optimizer=optimizer, exp_name=args.exp_name, batch_size=args.batch_size,
                   dataset_size=args.dataset_size, log_interval=args.log_interval, 
                   device=device, rnn_history=args.history)

