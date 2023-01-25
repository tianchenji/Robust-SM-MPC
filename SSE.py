# ---------------------------------------------------------------------------
# Set-Membership State Estimation
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2020-08-30
# ---------------------------------------------------------------------------

import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import e_project, sample_from_ellipsoid

class SSE:

    def __init__(self, A, B, D, C, Q, R, Sigma_init):
        '''
        initialize parameters for the LTI system
        Inputs:     A, B, D, C: system dynamics
                    Q, R: energy constraints of state disturbance and output disturbance respectively
        '''

        self.A = A
        self.B = B
        self.D = D
        self.C = C
        self.Q = Q
        self.R = R

        self.Sigma_init = Sigma_init

        self.beta, self.rho, self.Sigma_ss = self.find_sse_parms_opt()

    def sse_update(self, u, y, Sigma, xhat, delta):
        '''
        set-membership state estimation algorithm for instantaneous constraint
        sse_update returns state estimate xhat, ellipsoid shape matrix Sigma and shrinkage delta at time k
        Inputs:     u: control input at time k - 1
                    y: measurement at time k
                    Sigma: a posteri error covariance at time k - 1
                    xhat: a posteri state estimate at time k - 1
                    delta: shrinkage at time k - 1
        Outputs:    Sigma: a posteri error covariance at time k
                    xhat: a posteri state estimate at time k
                    delta: shrinkage at time k
        '''

        # for readability
        beta = self.beta
        rho  = self.rho

        # update Sigma
        Sigma_priori  = (1/(1 - beta))*np.dot(self.A, np.dot(Sigma, self.A.T)) + (1/beta)*np.dot(self.D, np.dot(self.Q, self.D.T))
        Sigma_posteri = np.linalg.inv((1 - rho)*np.linalg.inv(Sigma_priori) + rho*np.dot(self.C.T, np.dot(np.linalg.inv(self.R), self.C)))

        # update xhat
        IM = y - np.dot(self.C, (np.dot(self.A, xhat) + np.dot(self.B, u)))
        xhat = np.dot(self.A, xhat) + np.dot(self.B, u) + rho*np.dot(Sigma_posteri, np.dot(self.C.T, np.dot(np.linalg.inv(self.R), IM)))

        # update shrinkage delta
        IS = np.linalg.inv((1 / (1 - rho)) * np.dot(self.C, np.dot(Sigma_priori, self.C.T)) + (1 / rho) * self.R)
        delta = (1 - beta) * (1 - rho) * delta + np.dot(IM.T, np.dot(IS, IM))

        return (Sigma_posteri, xhat, delta)

    def sse_ss(self, beta, rho, Sigma):
        '''
        sse_ss returns steady-state error shape matrix
        Inputs:     Sigma: initial constraints of state estimate error
        Outputs:    Sigma: steady-state error shape matrix
        '''
        tol = 1e-8
        
        # compute steady state shape matrix by iteration
        while True:
            Sigma_prev   = Sigma
            Sigma_priori = (1/(1 - beta))*np.dot(self.A, np.dot(Sigma, self.A.T)) + (1/beta)*np.dot(self.D, np.dot(self.Q, self.D.T))
            Sigma        = np.linalg.inv((1 - rho)*np.linalg.inv(Sigma_priori) + rho*np.dot(self.C.T, np.dot(np.linalg.inv(self.R), self.C)))
            if np.linalg.norm(Sigma - Sigma_prev) <= tol:
                break

        return Sigma

    def find_sse_parms_opt(self):
        '''
        find_sse_parms_opt returns the optimal filtering parameters
        Outputs:    beta, rho: optimal filtering parameters
                    Sigma_ss: optimal steady-state error shape matrix
        '''

        # grid search
        print("searching optimal filtering parameters...")
        parm_trace_opt = []
        for beta in np.arange(0.1, 1.0, 0.1):
            for rho in np.arange(0.1, 1.0, 0.1):
                Sigma_ss = self.sse_ss(beta, rho, self.Sigma_init)
                if len(parm_trace_opt) == 0 or np.trace(Sigma_ss) < np.trace(parm_trace_opt[2]):
                    parm_trace_opt = [beta, rho, Sigma_ss]

        print("optimal filtering parameters found [beta, rho]: [{:.2f}, {:.2f}]".format(parm_trace_opt[0], parm_trace_opt[1]))

        return parm_trace_opt

if __name__ == '__main__':
    # bivariate example
    A     = np.array([[0.8, 1.0], [0.0, 0.9]])
    B     = np.array([[0], [1]])
    D     = np.array([[1, 0], [0, 1]])
    C     = np.array([[1, 0]])
    u     = np.array([[0.01]])

    x     = np.array([[-6.7], [1.4]]) # real value
    Sigma = np.array([[0.0005, 0], [0, 0.0036]])
    xhat  = np.array([[-6.69], [1.35]]) # initial guess
    delta = 0

    Q     = np.array([[2*0.1**2, 0], [0, 2*0.1**2]])
    R     = np.array([[0.05**2]])
    dim_w = len(Q)
    dim_v = len(R)

    n_iter          = 50
    xreal           = [0] * n_iter
    xreal[0]        = x
    # we visualize the first state
    vis_index       = 1
    xv_hat          = [0] * n_iter
    xv_real         = [0] * n_iter
    xv_lowerbound   = [0] * n_iter
    xv_upperbound   = [0] * n_iter
    xv_real[0]      = float(x[vis_index])

    sse = SSE(A, B, D, C, Q, R, Sigma)

    # sample disturbances
    w = sample_from_ellipsoid(n_iter, np.linalg.inv(Q), np.array([0.0, 0.0]))
    v = sample_from_ellipsoid(n_iter, np.linalg.inv(R), np.array([0.0]))

    for i in range(1, n_iter):
        xreal[i] = np.dot(A, xreal[i - 1]) + np.dot(B, u) + np.dot(D, w[i - 1].reshape((dim_w, 1)))
        xv_real[i] = float(xreal[i][vis_index])
        y = np.dot(C, xreal[i]) + v[i].reshape((dim_v, 1))
        Sigma, xhat, delta = sse.sse_update(u, y, Sigma, xhat, delta)
        s_min, s_max = e_project(np.linalg.inv(Sigma), xhat, delta)
        xv_lowerbound[i] = s_min[vis_index]
        xv_upperbound[i] = s_max[vis_index]
        xv_hat[i] = float(xhat[vis_index])

    xv_real = xv_real[1:n_iter]
    xv_hat  = xv_hat[1:n_iter]
    xv_lowerbound = xv_lowerbound[1:n_iter]
    xv_upperbound = xv_upperbound[1:n_iter]
    plt.figure()
    plt.plot(xv_hat, 'b*-', label='a posteri estimate')
    plt.plot(xv_real, 'g.-', label='real states')
    plt.plot(xv_lowerbound, 'r.-', label='lowerboud of state estimate')
    plt.plot(xv_upperbound, 'r.-', label='upperboud of state estimate')
    plt.legend()
    plt.grid()
    plt.show()