import argparse
# from script.interpolate import interp1d
import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from rossler_map import RosslerMap
from models import My_Model
from TP import lyapunov_exponent, newton


import scipy
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
class Rossler_model:
    def __init__(self, delta_t=1e-2, model=None, steps=10000, scale=False):

        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(steps//self.delta_t)

        self.rosler_nn = model
        self.initial_condition = np.array(value.init)
        self.init = value.init
        self.scaler = scaler
        self.scale = scale

    def full_traj_rnn(self, ):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.

        ROSSLER_MAP = RosslerMap(delta_t=self.delta_t)
        traj, t = ROSSLER_MAP.full_traj(rnn_history, self.init)
        if self.scale:
            traj = self.scaler.fit_transform(traj)

        h = self.rosler_nn.init_hidden(1)
        traj_tensor = torch.from_numpy(traj).float()

        w = traj_tensor[:rnn_history, 1]  # (history, 1)
        w = w.view((1, w.shape[0], 1))
        # w = torch.from_numpy(self.initial_condition).float()
        traj = []
        for i in tqdm(range(self.nb_steps)):

            w_new, h = self.rosler_nn(w, h)
            traj.append(w_new[0].item())
            w[:, -1, :] = w_new

        y = np.array(traj)

        # if your delta_t is different to 1e-2 then interpolate y
        # in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        return y

    def full_traj(self,):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 

        w = torch.from_numpy(self.initial_condition).float()
        traj = []
        for i in tqdm(range(self.nb_steps)):
            w_new = self.rosler_nn(w)
            traj.append(w_new[1].item())
            w = w_new

        y = np.array(traj)

        #if your delta_t is different to 1e-2 then interpolate y
        #in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        return y

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
          
        np.save('traj.npy',y)

def plot_gt_sample(value):

    Niter = int(value.steps // value.delta_t)
    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array(value.init)
    traj, t = ROSSLER_MAP.full_traj(Niter, INIT)
    y = traj[:, 1]
    fig = plt.figure(10)
    plt.title('Ground truth map')

    plt.plot(y)
    return y
    # plt.show()

def plot_3d(y, i, value):

    Niter = int(value.steps // value.delta_t)
    ROSSLER_MAP = RosslerMap(delta_t=value.delta_t)
    INIT = np.array(value.init)
    traj, t = ROSSLER_MAP.full_traj(Niter, INIT)
    if value.scale:
        traj = scaler.fit_transform(traj)

    fig1 = plt.figure(i)
    ax = fig1.gca(projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])

    fig2 = plt.figure(i+1)
    ax = fig2.gca(projection='3d')
    ax.plot(traj[:, 0], y, traj[:, 2])



def compute_stats(x, x_gt, name, dt=1e-2):

    
    plt.figure('Histogram')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of ' + name)
    # plt.text(23, 45, r'$\mu=15, b=3$')
    plt.hist(x=x, bins='auto', alpha=0.7, rwidth=0.85, label=name)


    plt.title('Histogram')
    # plt.text(23, 45, r'$\mu=15, b=3$')
    plt.hist(x=x_gt, bins='auto', alpha=0.7, rwidth=0.85, label='gt')
    plt.legend()
    
    
    plt.figure("CDF")
    plt.title("CDF")
    norm_cdf = scipy.stats.norm.cdf(x)  # calculate the cdf - also discrete
    plt.plot(x, norm_cdf, label=name)
    plt.legend()

    norm_cdf_gt = scipy.stats.norm.cdf(x_gt)  # calculate the cdf - also discrete
    plt.plot(x_gt, norm_cdf_gt, label='gt')
    plt.legend()

    # scipy.stats.pearsonr(x, y)[0]  # Pearson's r
    # scipy.stats.spearmanr(x, y)[0]   # Spearman's rho
    # scipy.stats.kendalltau(x, y)[0]   # Kendall's tau

    ### Auto correlation
    plt.figure("Auto correlation")
    plt.title("Auto correlation")

    auto = pd.plotting.autocorrelation_plot(x)
    auto.plot(labels=name)
    auto.legend()

    auto = pd.plotting.autocorrelation_plot(x_gt)
    auto.plot(labels='gt')
    auto.legend()

    ### Spectral density
    plt.figure("Spectral density")
    plt.title("Spectral density")

    ps = np.abs(np.fft.fft(x)) ** 2
    freqs = np.fft.fftfreq(x.size, dt)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], ps[idx], label=name)
    plt.legend()

    ps_gt = np.abs(np.fft.fft(x_gt)) ** 2
    freqs_gt = np.fft.fftfreq(x_gt.size, dt)
    idx_gt = np.argsort(freqs_gt)
    plt.plot(freqs_gt[idx_gt], ps_gt[idx_gt], label='gt')
    plt.legend()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--init", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    parser.add_argument("--model_path", type=str, default='lossv0model_0.pth')
    parser.add_argument("--steps", type=float, default=100)
    parser.add_argument('--delta_t', type=str, default=1e-2)
    parser.add_argument("--rnn", type=bool, default=False)
    parser.add_argument("--scale", type=bool, default=True)

    value = parser.parse_args()

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if is_cuda else torch.device("cpu")

    Model = My_Model()
    checkpoint = torch.load(value.model_path, map_location=device)
    Model.load_state_dict(checkpoint)

    ROSSLER = Rossler_model( delta_t=value.delta_t, model=Model, 
                             steps=value.steps, scale=value.scale)
    if value.rnn:
        y = ROSSLER.full_traj_rnn()
    else:
        y = ROSSLER.full_traj()

    # Plotting ground truth
    y_gt = plot_gt_sample(value)

    # Compute some statistics
    compute_stats(y, y_gt, name=value.model_path, dt=value.delta_t)

    # Plotting results 3D
    plot_3d(y, 30, value)

    # save output trajectory
    ROSSLER.save_traj(y)
    
    # plotting the generated trajectory
    fig = plt.figure("generated trajectory y-coordinate")
    plt.title('Map generated by the model' + value.model_path)
    plt.plot(y)
    
    # we used x and z from the exact model and we focused on our generated y
    Niter = int(value.steps//value.delta_t)
    ROSSLER_MAP = RosslerMap(delta_t=value.delta_t)
    
    ROSSLER_MAP = RosslerMap(delta_t=value.delta_t)
    traj, t = ROSSLER_MAP.full_traj(Niter, np.array(value.init))
    traj[:,1] = y
    ##
    
    # Compute the Equilibrium Point of the generated map
    fix_point = newton(Model, Model.batch_model_jacobian, np.array(value.init))
    
    error = norm(fix_point - ROSSLER_MAP.equilibrium())
    print("equilibrium state :", fix_point, ", error : ", error)
    print()
    
    # Compute the Lyapunov Exponent
    lyap = lyapunov_exponent(traj, Model.batch_model_jacobian, max_it=Niter, 
                                   delta_t=value.delta_t)
    print("Lyapunov Exponents :", lyap, "with delta t =", value.delta_t)

    plt.show()
