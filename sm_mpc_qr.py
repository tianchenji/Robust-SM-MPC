# ---------------------------------------------------------------------------
# Robust Output Feedback MPC -- Set Membership Approach
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2022-03-20
# ---------------------------------------------------------------------------

import sys
import copy
import pickle
import matplotlib.pyplot as plt

from casadi import *

from SSE import SSE
from sm_mpc import SM_MPC
from utils import sample_from_ellipsoid

if __name__ == '__main__':
    traj_random    = []
    control_random = []

    for num_of_samples in range(5):

        np.random.seed(num_of_samples*500)

        # system dynaimcs
        A = np.array([[1.0, 0.0, 0.0,    0.0, -0.1960, 0.0, 0.2, 0.0, 0.0,    0.0, -0.0131, 0.0],
                      [0.0, 1.0, 0.0, 0.1960,     0.0, 0.0, 0.0, 0.2, 0.0, 0.0131,     0.0, 0.0],
                      [0.0, 0.0, 1.0,    0.0,     0.0, 0.0, 0.0, 0.0, 0.2,    0.0,     0.0, 0.0],
                      [0.0, 0.0, 0.0,    1.0,     0.0, 0.0, 0.0, 0.0, 0.0,    0.2,     0.0, 0.0],
                      [0.0, 0.0, 0.0,    0.0,     1.0, 0.0, 0.0, 0.0, 0.0,    0.0,     0.2, 0.0],
                      [0.0, 0.0, 0.0,    0.0,     0.0, 1.0, 0.0, 0.0, 0.0,    0.0,     0.0, 0.2],
                      [0.0, 0.0, 0.0,    0.0, -1.9600, 0.0, 1.0, 0.0, 0.0,    0.0, -0.1960, 0.0],
                      [0.0, 0.0, 0.0, 1.9600,     0.0, 0.0, 0.0, 1.0, 0.0, 0.1960,     0.0, 0.0],
                      [0.0, 0.0, 0.0,    0.0,     0.0, 0.0, 0.0, 0.0, 1.0,    0.0,     0.0, 0.0],
                      [0.0, 0.0, 0.0,    0.0,     0.0, 0.0, 0.0, 0.0, 0.0,    1.0,     0.0, 0.0],
                      [0.0, 0.0, 0.0,    0.0,     0.0, 0.0, 0.0, 0.0, 0.0,    0.0,     1.0, 0.0],
                      [0.0, 0.0, 0.0,    0.0,     0.0, 0.0, 0.0, 0.0, 0.0,    0.0,     0.0, 1.0]])
        B = np.array([[    0.0,     0.0, -0.2816,  0.0],
                      [    0.0,  0.2816,     0.0,  0.0],
                      [-0.0400,     0.0,     0.0,  0.0],
                      [    0.0,  8.6207,     0.0,  0.0],
                      [    0.0,     0.0,  8.6207,  0.0],
                      [    0.0,     0.0,     0.0,  5.0],
                      [    0.0,     0.0, -5.6322,  0.0],
                      [    0.0,  5.6322,     0.0,  0.0],
                      [-0.4000,     0.0,     0.0,  0.0],
                      [    0.0, 86.2069,     0.0,  0.0],
                      [    0.0,     0.0, 86.2069,  0.0],
                      [    0.0,     0.0,     0.0, 50.0]])
        C = np.identity(len(A))
        K = - np.array([[    0.0,    0.0, -0.9813,    0.0,    0.0,    0.0,     0.0,    0.0, -1.3944,    0.0,    0.0,    0.0],
                        [    0.0, 0.0068,     0.0, 0.0588,    0.0,    0.0,     0.0, 0.0113,     0.0, 0.0168,    0.0,    0.0],
                        [-0.0068,    0.0,     0.0,    0.0, 0.0588,    0.0, -0.0113,    0.0,     0.0,    0.0, 0.0168,    0.0],
                        [    0.0,    0.0,     0.0,    0.0,    0.0, 0.0182,     0.0,    0.0,     0.0,    0.0,    0.0, 0.0218]])

        # state and input constraints
        F = np.array([[0.0, 0.0, 0.0, -1.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,  1.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,  0.0, -1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,  0.0,  1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,  0.0,  0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,  0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        G = np.array([[ 0.0, 0.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.0, 0.0],
                      [ 1.0, 0.0, 0.0, 0.0]])
        f = np.array([[pi/9],
                      [pi/9],
                      [pi/9],
                      [pi/9],
                      [pi/9],
                      [pi/9],
                      [5.0],
                      [5.0]])

        # energy bounds on state and output disturbances
        state_disturb  = np.array([0.03]*len(A))
        output_disturb = np.array([0.03]*len(A))

        Q     = np.diag(state_disturb**2)
        R     = np.diag(output_disturb**2)
        dim_w = len(Q)
        dim_v = len(R)
        sigma = np.diag(state_disturb**2)

        # define the state estimator
        sse   = SSE(A, B, np.identity(len(A)), C, Q, R, sigma)
        sigma = copy.deepcopy(sse.Sigma_ss)

        # initial guess
        xhat = np.array([[5.0],
                         [4.0],
                         [0.0],
                         [-0.03],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0]])

        # real initial state
        x = np.array([[5.0],
                      [4.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0]])

        # mRPI parameters
        r = 100

        # prediction horizon
        N = 15

        sm_mpc = SM_MPC(A, B, F, G, f, Q, K, r, N, sse)

        xbar = copy.deepcopy(xhat)

        # visualize the closed-loop trajectory and the control inputs
        xbar_vis = []
        x_vis    = []
        ubar_vis = []
        u_vis    = []

        threshold = 0.2

        # keep iterating until the nominal state converges to 0
        delta     = 0
        time_step = 0
        while True:

            # compute optimal control
            sol, solver_status = sm_mpc.solve(xbar, time_step)

            if solver_status != "Solve_Succeeded":
                print("MPC failed. Control terminated. Error:", solver_status)
                break

            xbar     = np.array(sol["x"][sm_mpc.first_state_index.xbar[0]:sm_mpc.first_state_index.ubar[0]:sm_mpc.horizon])
            ubar_opt = np.array(sol["x"][sm_mpc.first_state_index.ubar[0]::(sm_mpc.horizon - 1)])
            u_opt = np.dot(K, (xhat - xbar)) + ubar_opt

            xbar_vis.append(xbar.flatten())
            x_vis.append(x.flatten())
            ubar_vis.append(ubar_opt.flatten())
            u_vis.append(u_opt.flatten())

            w = sample_from_ellipsoid(1, np.linalg.inv(Q), np.zeros(dim_w))
            v = sample_from_ellipsoid(1, np.linalg.inv(R), np.zeros(dim_v))

            # simulate forward
            x    = np.dot(A, x) + np.dot(B, u_opt) + w.reshape((dim_w, 1))
            xbar = np.dot(A, xbar) + np.dot(B, ubar_opt)

            # estimate current states
            y = np.dot(C, x) + v.reshape((dim_v, 1))

            sigma, xhat, delta = sse.sse_update(u_opt, y, sigma, xhat, delta)

            print("The norm of the state estimate is:", np.linalg.norm(xhat))

            if np.linalg.norm(xhat) <= threshold:
                print("The system state has reached the origin.")
                break

            time_step += 1

        traj_random.append(x_vis)
        control_random.append(u_vis)

    # save sample paths with random noise
    min_len = min([len(sample) for sample in traj_random])

    traj_random    = [sample[:min_len] for sample in traj_random]
    traj_random    = np.array(traj_random)
    control_random = [sample[:min_len] for sample in control_random]
    control_random = np.array(control_random)
    
    # save nominal path
    traj_nominal    = np.array(xbar_vis[:min_len])
    control_nominal = np.array(ubar_vis[:min_len])

    # save tightened constraint
    tightened_cstr = f.flatten() - np.array(sm_mpc.tightened_constraints[:min_len])

    pickle.dump([traj_random, control_random, traj_nominal, control_nominal, tightened_cstr],
                 open("results/quadrotor_sm_mpc.p", "wb"))