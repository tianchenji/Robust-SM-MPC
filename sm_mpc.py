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
from scipy.linalg import solve_discrete_lyapunov

from SSE import SSE
from utils import FirstStateIndex, sample_from_ellipsoid

class SM_MPC:

    def __init__(self, A, B, F, G, f, Q, K, r, N, sse):
        '''
        initialize parameters for the robust output feedback MPC
        Note: the variable "self.horizon" includes the current time step
        Inputs:     A, B: system dynamics
                    F, G, f: state and input constriants
                    Q: ellipsoidal constraints on state disturbances
                    K: fixed feedback control matrix
                    r: parameters in mRPI approximation
                    N: the prediction horizon
                    sse: an instance of the SSE class
        '''

        self.A, self.B         = A, B
        self.F, self.G, self.f = F, G, f.flatten()
        self.Q                 = Q
        self.K                 = K
        self.r                 = r
        self.horizon           = N + 1
        self.sse               = sse

        self.A_K = self.A + np.dot(self.B, self.K)
        self.psi = self.F + np.dot(self.G, self.K)

        self.dim_x = len(self.A)
        self.dim_h = len(self.F)
        self.dim_u = np.shape(self.B)[1]

        tightened_cstr_tol = 1e-6

        # compute tightened constraints
        self.tightened_constraints = []

        h_cache   = None
        time_step = 0
        while True:
            h_cur, h_cache = self.get_tightened_constraint(h_cache, time_step)

            if time_step == 0 or (h_cur - self.tightened_constraints[-1] > tightened_cstr_tol).any() == True:
                self.tightened_constraints.append(h_cur)
                time_step += 1
            else:
                break

        assert (h_cur > self.tightened_constraints[-1]).all() == True
        self.tightened_constraint_inf = h_cur + 100 * (h_cur - self.tightened_constraints[-1])

        print("The tightened constraint is approximated since time {}".format(len(self.tightened_constraints)))

        print("The constraint is at most tightened by: {}".format(np.round(self.tightened_constraint_inf, decimals=6)))
        if (self.tightened_constraint_inf < self.f).all() == False:
            print("The tightened constraint is too restrictive to be satisfied. The robustness cannot be achieved.")
            sys.exit()
        else:
            self.nu = self.MRPI()
            print("The smallest positive integer nu for the MRPI set is: {:d}".format(self.nu))

        # define the parameters for the MPC cost function
        self.Q_cost = np.eye(self.dim_x)
        self.R_cost = 0.01 * np.eye(self.dim_u)
        self.P      = self.get_trm_cost()

        self.first_state_index = FirstStateIndex(A=A, B=B, N=N)

        # number of optimization variables
        self.num_of_x = self.dim_x * self.horizon + self.dim_u * (self.horizon - 1)
        self.num_of_g = self.dim_x * self.horizon + self.dim_h * (self.horizon + self.nu)

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

    def MRPI(self):
        '''
        MRPI computes the maximal positively invariant set for system xbar = (A + BK)xbar under the tightened constraints
        Outputs:    nu: the smallest positive integer used for describing the MRPI set
        '''

        opts                      = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"]        = 0

        # define optimization variables
        x = SX.sym('x', self.dim_x)

        # solve nu
        l_vec = np.zeros(self.dim_h)

        n = 0
        while True:
            inter_var = np.dot(self.psi, np.linalg.matrix_power(self.A_K, n + 1))
            cost_vec  = - mtimes(inter_var, x)
            g         = [None] * (n + 1)
            for i in range(n + 1):
                inter_var = np.dot(self.psi, np.linalg.matrix_power(self.A_K, i))
                g[i]      = mtimes(inter_var, x)

            ubg = list(self.f - self.tightened_constraint_inf) * (n + 1)

            for row_idx in range(self.dim_h):
                nlp    = {'x':x, 'f':cost_vec[row_idx], 'g':vertcat(*g)}
                solver = nlpsol('solver', 'ipopt', nlp, opts)
                res    = solver(ubg=ubg)

                l_vec[row_idx] = float(- res['f'])

            if (l_vec <= self.f - self.tightened_constraint_inf).all() == True:
                nu = n
                break
            else:
                n += 1

        return nu

    def get_trm_cost(self):
        '''
        get_trm_cost returns the matrix P associated with the terminal cost
        Outputs:    P: the matrix associated with the terminal cost in the objective
        '''

        A_lyap = (self.A + np.dot(self.B, self.K)).T
        Q_lyap = self.Q_cost + np.dot(self.K.T, np.dot(self.R_cost, self.K))

        P = solve_discrete_lyapunov(A_lyap, Q_lyap)

        return P

    def solve(self, xbar0, time_step):
        '''
        solve returns optimal control sequences
        Inputs:     xbar0: initial nominal state
        Outputs:    res: optimal solution to the finite time MPC problem
        '''

        # construct tightened constraints
        h_cur_tv = self.tightened_constraints[time_step:]
        h_cur_ti = [self.tightened_constraint_inf] * (self.horizon - 1 - len(h_cur_tv))
        h_cur    = h_cur_tv + h_cur_ti

        opts                      = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"]        = 0

        xbar = [0] * self.horizon
        ubar = [0] * (self.horizon - 1)

        ineq_cons_idx = self.dim_x * self.horizon

        # define optimization variables
        x = SX.sym('x', self.num_of_x)
        
        # initialize optimization variables
        x0 = [0] * self.num_of_x
        for i in range(self.dim_x):
            x0[self.first_state_index.xbar[i]] = xbar0[i]

        # define lowerbounds and upperbounds for g constraints
        g_lowerbound = [0] * self.num_of_g
        g_upperbound = [0] * self.num_of_g

        for i in range(self.dim_x):
            g_lowerbound[i] = xbar0[i]
            g_upperbound[i] = xbar0[i]

        for i in range(ineq_cons_idx, self.num_of_g):
            g_lowerbound[i] = -1e10

        for i in range(self.horizon - 1):
            g_upperbound[ineq_cons_idx + i*self.dim_h:ineq_cons_idx + (i + 1)*self.dim_h] = self.f - h_cur[i]

        # upperbound for the terminal constraints
        trm_cons = list(self.f - self.tightened_constraint_inf)*(self.nu + 1)
        g_upperbound[ineq_cons_idx + (self.horizon - 1)*self.dim_h:self.num_of_g] = trm_cons

        # define cost functions
        cost = 0.0

        # store nominal states and inputs at each time step for readability
        for i in range(self.horizon):
            xbar[i] = x[self.first_state_index.xbar[0] + i:self.first_state_index.ubar[0]:self.horizon]
        for i in range(self.horizon - 1):
            ubar[i] = x[self.first_state_index.ubar[0] + i::(self.horizon - 1)]

        # penalty on the nominal state xbar
        for i in range(self.horizon - 1):
            cost += mtimes([xbar[i].T, self.Q_cost, xbar[i]])

        # penalty on the terminal state xbar_N
        cost += mtimes([xbar[self.horizon - 1].T, self.P, xbar[self.horizon - 1]])

        # penalty on control inputs
        for i in range(self.horizon - 1):
            cost += mtimes([ubar[i].T, self.R_cost, ubar[i]])

        # define g constraints
        g = [None] * (self.horizon + self.horizon + self.nu)

        # equality constraints
        g[0] = xbar[0]
        
        for i in range(self.horizon - 1):
            g[1 + i] = xbar[i + 1] - mtimes(self.A, xbar[i]) - mtimes(self.B, ubar[i])

        # inequality constraints
        for i in range(self.horizon - 1):
            g[self.horizon + i] = mtimes(self.F, xbar[i]) + mtimes(self.G, ubar[i])

        for i in range(self.nu + 1):
            inter_var = np.dot(self.psi, np.linalg.matrix_power(self.A_K, i))
            g[self.horizon + self.horizon - 1 + i] = mtimes(inter_var, xbar[self.horizon - 1])

        # create the NLP
        nlp = {'x':x, 'f':cost, 'g':vertcat(*g)}

        solver = nlpsol('solver', 'ipopt', nlp, opts)
        
        res = solver(x0=x0, lbg=g_lowerbound, ubg=g_upperbound)
        return res, solver.stats()["return_status"]

if __name__ == '__main__':
    traj_random    = []
    control_random = []

    for num_of_samples in range(5):

        np.random.seed(num_of_samples*700)

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
        dim_w = len(Q)
        dim_v = len(R)
        sigma = np.array([[state_disturb**2, 0], [0, state_disturb**2]])

        # define the state estimator
        sse   = SSE(A, B, np.identity(len(A)), C, Q, R, sigma)
        sigma = copy.deepcopy(sse.Sigma_ss)

        # initial guess
        xhat = np.array([[-3.0],[-8.0]])

        # real initial state
        x = np.array([[-3.1],[-8.0]]) # specially designed for this example

        r = 6

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

            if np.linalg.norm(xhat) <= threshold and time_step >= 17:
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

    pickle.dump([traj_random, control_random, traj_nominal, control_nominal],
                 open("results/double_integrator_sm_mpc.p", "wb"))