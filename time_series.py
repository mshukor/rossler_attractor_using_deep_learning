import argparse
# from script.interpolate import interp1d
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from rossler_map import RosslerMap
from train import model

import seaborn as sns
import scipy
import pandas as pd

class Rossler_model:
    def __init__(self, delta_t=1e-2, model=None, steps=10000):

        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(steps//self.delta_t)

        self.rosler_nn = model
        self.initial_condition = np.array(value.init)

    def full_traj(self,):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 

        w = torch.from_numpy(self.initial_condition).float()
        traj = []
        for i in tqdm(range(self.nb_steps)):
            w_new = model(w)
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


def compute_stats(x, x_gt, i, name, dt=1e-2):

    plt.figure(i)
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

    plt.figure(i + 1)
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
    plt.figure(i + 2)
    plt.title("Auto correlation")

    auto = pd.plotting.autocorrelation_plot(x)
    auto.plot(labels=name)
    auto.legend()

    auto = pd.plotting.autocorrelation_plot(x_gt)
    auto.plot(labels='gt')
    auto.legend()

    ### Spectral density
    plt.figure(i + 3)
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
    parser.add_argument("--model_path", type=str, default='basicmodel_360.pth')
    parser.add_argument("--steps", type=float, default=100)
    parser.add_argument('--delta_t', type=str, default=1e-2)

    value = parser.parse_args()


    # model = nn.Sequential(nn.Linear(3, 10),
    #                        nn.ReLU(),
    #                        nn.Linear(10, 5),
    #                        nn.ReLU(),
    #                        nn.Linear(5, 3))

    checkpoint = torch.load(value.model_path)
    model.load_state_dict(checkpoint)


    ROSSLER = Rossler_model(delta_t=value.delta_t, model=model, steps=value.steps)

    y = ROSSLER.full_traj()

    y_gt = plot_gt_sample(value)

    compute_stats(y, y_gt, 0, name=value.model_path, dt=value.delta_t)


    fig = plt.figure(20)
    # ax = fig.gca(projection='3d')
    # ax.plot(y[:, 0], y[:, 1], y[:, 2])

    # fix_point = newton(ROSSLER_MAP.v_eq, ROSSLER_MAP.jacobian, INIT)
    #
    # error = norm(fix_point - ROSSLER_MAP.equilibrium())
    # print("equilibrium state :", fix_point, ", error : ", error)

    # lyap = lyapunov_exponent(traj, ROSSLER_MAP.jacobian, max_it=Niter, delta_t=delta_t)
    # print("Lyapunov Exponents :", lyap, "with delta t =", delta_t)

    ROSSLER.save_traj(y)
    plt.title('Map generated by the model' + value.model_path)
    plt.plot(y)
    plt.show()

