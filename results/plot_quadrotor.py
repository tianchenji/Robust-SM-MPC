import pickle
import numpy as np
import matplotlib.pyplot as plt

from math import pi

plt.rcParams.update({
    'font.size': 22,
    'font.family': "Times New Roman",
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

if __name__ == '__main__':

    #---------------------------------------------#
    # figure 3: closed-loop response of quadrotor #
    #---------------------------------------------#

    cl_sm_mpc = pickle.load(open("quadrotor_sm_mpc.p", "rb"))
    traj_random, control_random, traj_nominal, control_nominal, tightened_cstr = cl_sm_mpc

    # plot phi
    plt.figure(figsize=(8.5,7))
    for sample_traj in traj_random:
        plt.plot(sample_traj[:,3], color='red')
    plt.plot(-tightened_cstr[:,0], color='blue', linewidth=3.5)
    plt.plot(tightened_cstr[:,1], color='blue', linewidth=3.5)
    plt.axhline(y=pi/9, color='green', linestyle='-', linewidth=3.5)
    plt.axhline(y=-pi/9, color='green', linestyle='-', linewidth=3.5)
    plt.plot(traj_nominal[:,3], color='black', linewidth=3.5)

    plt.xlabel(r'time $k$')
    plt.ylabel(r'state $\phi$')
    plt.xlim(0, len(traj_random[0])-1)
    plt.ylim(-0.38, 0.38)
    plt.xticks(np.arange(0, 21, 5))

    # plot theta
    plt.figure(figsize=(8.5,7))
    for sample_traj in traj_random:
        plt.plot(sample_traj[:,4], color='red')
    plt.plot(-tightened_cstr[:,2], color='blue', linewidth=3.5)
    plt.plot(tightened_cstr[:,3], color='blue', linewidth=3.5)
    plt.axhline(y=pi/9, color='green', linestyle='-', linewidth=3.5)
    plt.axhline(y=-pi/9, color='green', linestyle='-', linewidth=3.5)
    plt.plot(traj_nominal[:,4], color='black', linewidth=3.5)

    plt.xlabel(r'time $k$')
    plt.ylabel(r'state $\theta$')
    plt.xlim(0, len(traj_random[0])-1)
    plt.ylim(-0.38, 0.38)
    plt.xticks(np.arange(0, 21, 5))

    # plot p_x
    plt.figure(figsize=(8.5,7))
    for sample_traj in traj_random:
        plt.plot(sample_traj[:,0], color='red')
    plt.plot(traj_nominal[:,0], color='black', linewidth=3.5)

    plt.xlabel(r'time $k$')
    plt.ylabel(r'state $p_x$')
    plt.xlim(0, len(traj_random[0])-1)
    plt.ylim(0, 5.2)
    plt.xticks(np.arange(0, 21, 5))

    # plot p_y
    plt.figure(figsize=(8.5,7))
    for sample_traj in traj_random:
        plt.plot(sample_traj[:,1], color='red')
    plt.plot(traj_nominal[:,1], color='black', linewidth=3.5)

    plt.xlabel(r'time $k$')
    plt.ylabel(r'state $p_y$')
    plt.xlim(0, len(traj_random[0])-1)
    plt.ylim(0, 5)
    plt.xticks(np.arange(0, 21, 5))

    plt.show()