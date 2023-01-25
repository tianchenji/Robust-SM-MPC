# ---------------------------------------------------------------------------
# Robust Output Feedback MPC -- Set Membership Approach
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2020-03-06
# ---------------------------------------------------------------------------
import sys
import copy
import pickle
import matplotlib.pyplot as plt

from casadi import *

from SSE import SSE

class SM_MPC:

    def __init__(self, A, B, F, G, f, Q, K, r, sse):

        self.A, self.B         = A, B
        self.F, self.G, self.f = F, G, f.flatten()
        self.Q                 = Q
        self.K                 = K
        self.r                 = r
        self.sse               = sse

        self.A_K = self.A + np.dot(self.B, self.K)
        self.psi = self.F + np.dot(self.G, self.K)

    def get_ellipsoid_max(self, M, P):
        '''
        compute max Md where d is an ellipsoid parameterized by shape matrix P
        '''

        dim_d = len(P)
        dim_h = len(M)

        opts                      = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"]        = 0

        d = SX.sym('d', dim_d)

        cost_vec  = - mtimes([M, d])
        g         = mtimes([d.T, P, d])

        h_vec = np.zeros(dim_h)
        for row_idx in range(dim_h):
            nlp    = {'x':d, 'f':cost_vec[row_idx], 'g':g}
            solver = nlpsol('solver', 'ipopt', nlp, opts)
            res    = solver(x0=[0.0]*dim_d, ubg=1)

            h_vec[row_idx] = float(- res['f'])

        return h_vec

    def get_tightened_constraint(self, h_cache, time_step):

        if time_step == 0:
            h_cur = self.get_ellipsoid_max(self.F, np.linalg.inv(self.sse.Sigma_ss))
            h_cache = None

        else:
            # solve the s term in s dynamics
            inter_var = np.linalg.matrix_power(self.A_K, time_step)
            h_s_s     = self.get_ellipsoid_max(np.dot(self.psi, inter_var), np.linalg.inv(self.sse.Sigma_ss))

            # solve the w term in s dynamics
            inter_var = np.linalg.matrix_power(self.A_K, time_step - 1)
            h_s_w     = self.get_ellipsoid_max(np.dot(self.psi, inter_var), np.linalg.inv(self.Q))

            # solve the e term in s dynamics
            inter_var = np.dot(np.linalg.matrix_power(self.A_K, time_step - 1), np.dot(self.B, self.K))
            h_s_e     = self.get_ellipsoid_max(np.dot(self.psi, inter_var), np.linalg.inv(self.sse.Sigma_ss))

            if time_step == 1:
                h_e = self.get_ellipsoid_max(np.dot(self.G, self.K), np.linalg.inv(self.sse.Sigma_ss))

                h_cur   = h_s_s + h_s_w + h_s_e + h_e
                h_cache = h_s_w + h_s_e + h_e

            else:
                h_cur   = h_cache + h_s_s + h_s_w + h_s_e
                h_cache = h_cache + h_s_w + h_s_e

        return h_cur, h_cache

if __name__ == '__main__':
    # limit on disturbances
    state_disturb  = 0.2
    output_disturb = 0.1

    # system dynamics
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    B = np.array([[1.0],
                  [1.0]])
    C = np.array([[1.0, 1.0]])
    K = np.array([[-0.6136, -0.9962]])
    L = np.array([[-1.0],
                  [-1.0]])

    # state and input constraints
    F = np.array([[-1.0,0.0],
                  [1.0, 0.0],
                  [0.0, -1.0],
                  [0.0, 1.0],
                  [0.0, 0.0],
                  [0.0, 0.0]])
    G = np.array([[0.0],
                  [0.0],
                  [0.0],
                  [0.0],
                  [-1.0],
                  [1.0]])
    f = np.array([[50.0],
                  [3.0],
                  [50.0],
                  [3.0],
                  [3.0],
                  [3.0]])

    # energy bounds on state and output disturbances
    Q     = np.array([[state_disturb**2, 0], [0, state_disturb**2]])
    R     = np.array([[output_disturb**2]])
    sigma = np.array([[state_disturb**2, 0], [0, state_disturb**2]])

    # define the state estimator
    sse = SSE(A, B, np.identity(len(A)), C, Q, R, sigma)

    sigma = copy.deepcopy(sse.Sigma_ss)

    r = 6

    sm_mpc = SM_MPC(A, B, F, G, f, Q, K, r, sse)

    tightened_constraint = []

    h_cache = None

    for i in range(11):
        h_cur, h_cache = sm_mpc.get_tightened_constraint(h_cache, i)

        tightened_constraint.append(h_cur)

    tightened_constraint = np.array(tightened_constraint)
    
    pickle.dump(tightened_constraint, open("results/cstr_tightening_sm_mpc.p", "wb"))

    plt.plot(tightened_constraint[:,4], marker='.')
    plt.show()