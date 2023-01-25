import pickle
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 15,
    'font.family': "Times New Roman",
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

if __name__ == '__main__':

    #------------------------------------------------------------------------#
    # figure 1: time-varying constraint tightening with different approaches #
    #------------------------------------------------------------------------#

    cstr_tightening_two_tubes   = pickle.load(open("cstr_tightening_two_tubes.p", "rb"))
    cstr_tightening_single_tube = pickle.load(open("cstr_tightening_single_tube.p", "rb"))
    cstr_tightening_sm_mpc      = pickle.load(open("cstr_tightening_sm_mpc.p", "rb"))

    # plot constrint tightening for x1
    plt.figure()
    plt.plot(cstr_tightening_two_tubes[:,0], color='blue', marker='.', markersize=10)
    plt.plot(cstr_tightening_single_tube[:,0], color='black', marker='.', markersize=10)
    plt.plot(cstr_tightening_sm_mpc[:,0], color='red', marker='.', markersize=10)

    plt.xlim(0, len(cstr_tightening_sm_mpc)-1)
    plt.ylim(0, 2.7)
    plt.xlabel(r'time $k$')
    plt.ylabel(r'Size of constraint tightening')

    # plot constrint tightening for x2
    plt.figure()
    plt.plot(cstr_tightening_two_tubes[:,2], color='blue', marker='.', markersize=10)
    plt.plot(cstr_tightening_single_tube[:,2], color='black', marker='.', markersize=10)
    plt.plot(cstr_tightening_sm_mpc[:,2], color='red', marker='.', markersize=10)

    plt.xlim(0, len(cstr_tightening_sm_mpc)-1)
    plt.ylim(0, 2.7)
    plt.xlabel(r'time $k$')
    plt.ylabel(r'Size of constraint tightening')

    # plot constrint tightening for u
    plt.figure()
    plt.plot(cstr_tightening_two_tubes[:,4], color='blue', marker='.', markersize=10)
    plt.plot(cstr_tightening_single_tube[:,4], color='black', marker='.', markersize=10)
    plt.plot(cstr_tightening_sm_mpc[:,4], color='red', marker='.', markersize=10)

    plt.xlim(0, len(cstr_tightening_sm_mpc)-1)
    plt.ylim(0, 2.7)
    plt.xlabel(r'time $k$')
    plt.ylabel(r'Size of constraint tightening')

    plt.show()