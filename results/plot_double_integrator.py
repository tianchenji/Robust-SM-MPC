import pickle
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 26,
    'font.family': "Times New Roman",
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

if __name__ == '__main__':

    #-----------------------------------------------------#
    # figure 2: closed-loop response of double integrator #
    #-----------------------------------------------------#

    cl_single_tube = pickle.load(open("double_integrator_single_tube.p", "rb"))
    traj_random, control_random, traj_nominal, control_nominal = cl_single_tube

    # plot x1 and x2 with single tube approach
    for x_i in range(2):
        plt.figure(figsize=(9.3,8))
        for sample_traj in traj_random:
            plt.plot(sample_traj[:,x_i], color='red')
        plt.plot(traj_nominal[:, x_i], color='black', linewidth=4)
        plt.axhline(y=3, color='green', linestyle='-', linewidth=4)

        plt.xlabel(r'time $k$')
        plt.ylabel(r'state $x[{:d}]$'.format(x_i + 1))
        plt.xlim(0, len(traj_random[0])-1)
        if x_i == 0:
            plt.ylim(-18.2, 3.5)
        else:
            plt.ylim(-8, 4)

    # plot u with single tube approach
    plt.figure(figsize=(9.3,8))
    for sample_control in control_random:
        plt.plot(sample_control[:,0], color='red')
    plt.plot(control_nominal[:, 0], color='black', linewidth=4)
    plt.axhline(y=3, color='green', linestyle='-', linewidth=4)
    plt.axhline(y=-3, color='green', linestyle='-', linewidth=4)

    plt.xlim(0, len(control_random[0])-1)
    plt.ylim(-2.0, 3.2)
    plt.xlabel(r'time $k$')
    plt.ylabel(r'input $u$')


    cl_sm_mpc = pickle.load(open("double_integrator_sm_mpc.p", "rb"))
    traj_random, control_random, traj_nominal, control_nominal = cl_sm_mpc

    # plot x1 and x2 with single tube approach
    for x_i in range(2):
        plt.figure(figsize=(9.3,8))
        for sample_traj in traj_random:
            plt.plot(sample_traj[:,x_i], color='red')
        plt.plot(traj_nominal[:, x_i], color='black', linewidth=4)
        plt.axhline(y=3, color='green', linestyle='-', linewidth=4)

        plt.xlabel(r'time $k$')
        plt.ylabel(r'state $x[{:d}]$'.format(x_i + 1))
        plt.xlim(0, len(traj_random[0])-1)
        if x_i == 0:
            plt.ylim(-18.2, 3.5)
        else:
            plt.ylim(-8, 4)

    # plot u with single tube approach
    plt.figure(figsize=(9.3,8))
    for sample_control in control_random:
        plt.plot(sample_control[:,0], color='red')
    plt.plot(control_nominal[:, 0], color='black', linewidth=4)
    plt.axhline(y=3, color='green', linestyle='-', linewidth=4)
    plt.axhline(y=-3, color='green', linestyle='-', linewidth=4)

    plt.xlim(0, len(control_random[0])-1)
    plt.ylim(-2.0, 3.2)
    plt.xlabel(r'time $k$')
    plt.ylabel(r'input $u$')

    plt.show()