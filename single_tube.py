# ---------------------------------------------------------------------------
# Robust Output Feedback MPC -- Single Tube Approach
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2022-02-24
# Paper: M. Kogel and R. Findeisen. "Robust output feedback MPC for uncertain 
# linear systems with reduced conservatism." IFAC-PapersOnLine, 2017.
# ---------------------------------------------------------------------------

import sys
import copy
import pickle
import itertools as it
import matplotlib.pyplot as plt

from casadi import *
from math import sqrt
from scipy.spatial import ConvexHull
from scipy.linalg import solve_discrete_lyapunov

from utils import polytope_vertices, FirstStateIndex, sample_from_ellipsoid, bounding_box_ellipsoid

class MPC_single_tube:

    def __init__(self, A, B, C, F, G, f, Q, R, K, L, r, N):
        '''
        initialize parameters for the robust output feedback MPC
        Note: the variable "self.horizon" includes the current time step
        Inputs:     A, B, C: system dynamics
                    F, G, f: state and input constriants
                    Q, R: ellipsoidal constraints on state and output disturbances
                    K: fixed feedback control matrix
                    r: parameters in mRPI approximation
                    N: the prediction horizon
        '''

        self.A, self.B, self.C = A, B, C
        self.F, self.G, self.f = F, G, f.flatten()
        self.Q, self.R         = Q, R
        self.K, self.L         = K, L
        self.r                 = r
        self.horizon           = N + 1

        self.A_L = self.A + np.dot(self.L, self.C)
        self.A_K = self.A + np.dot(self.B, self.K)
        self.psi = self.F + np.dot(self.G, self.K)

        self.dim_x = len(self.A)
        self.dim_v = len(self.C)
        self.dim_h = len(self.F)
        self.dim_u = np.shape(self.B)[1]

        tightened_cstr_tol = 1e-6

        # construct dynamics for augmented error states
        F_dyn_row_1 = np.concatenate((self.A + np.dot(self.L, self.C), np.zeros_like(self.A)), axis=1)
        F_dyn_row_2 = np.concatenate((- np.dot(self.L, self.C), self.A + np.dot(self.B, self.K)), axis=1)
        self.F_dyn  = np.concatenate((F_dyn_row_1, F_dyn_row_2), axis=0)

        G_dyn_row_1 = np.concatenate((np.identity(len(self.L)), self.L), axis=1)
        G_dyn_row_2 = np.concatenate((np.zeros_like(self.A), - self.L), axis=1)
        self.G_dyn  = np.concatenate((G_dyn_row_1, G_dyn_row_2), axis=0)

        # overapproximate ellipsoidal disturbances by polytopes
        V_w = bounding_box_ellipsoid(np.eye(self.dim_x), np.linalg.inv(self.Q))
        V_v = bounding_box_ellipsoid(np.eye(self.dim_v), np.linalg.inv(self.R))

        self.w_vertices = polytope_vertices(V_w)
        if self.dim_v == 1:
            self.v_vertices = 1 / V_v
        else:
            self.v_vertices = polytope_vertices(V_v)

        self.wv_vertices = list(it.product(self.w_vertices, self.v_vertices))
        self.wv_vertices = np.array([np.concatenate((w, v), axis=0) for w, v in self.wv_vertices])

        # compute tightened constraints
        V_E_inf     = self.mRPI_e()
        V_E_inf_aug = np.concatenate((V_E_inf, np.zeros_like(V_E_inf)), axis=1)

        V_Xi     = np.concatenate((1e5*np.identity(len(self.A)), -1e5*np.identity(len(self.A))), axis=0)
        V_Xi_aug = np.concatenate((np.zeros_like(V_Xi), V_Xi), axis=1)

        V_Z_prior        = np.concatenate((V_E_inf_aug, V_Xi_aug), axis=0)
        Z_prior_vertices = polytope_vertices(V_Z_prior)

        tightened_cstr_prior       = self.get_tightened_constraint(V_Z_prior)
        self.tightened_constraints = [tightened_cstr_prior]

        while True:
            V_Z_post, Z_post_vertices = self.get_tube_k(Z_prior_vertices)

            tightened_cstr_post = self.get_tightened_constraint(V_Z_post)
            if (tightened_cstr_post - tightened_cstr_prior > tightened_cstr_tol).any() == True:
                self.tightened_constraints.append(tightened_cstr_post)
                tightened_cstr_prior = tightened_cstr_post
                Z_prior_vertices     = Z_post_vertices
            else:
                break

        assert (tightened_cstr_post > tightened_cstr_prior).all() == True
        self.tightened_constraint_inf = tightened_cstr_post + 100 * (tightened_cstr_post - tightened_cstr_prior)

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

    def mRPI_polytope(self, A, V):
        '''
        mRPI_polytope computes the minimal RPI set of a given dynamics e = Ae + d
        Assumption: the disturbance lives in a convex polytope
        Inputs:     A: dynamic matrix
                    V: description of the polytope in terms of linear inequalities Vd <= 1
        Outputs:    alpha: the scaling factor
        '''

        dim_d          = np.shape(V)[1]
        dim_cost_alpha = len(V)

        opts                      = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"]        = 0

        # solve alpha
        # define optimization variables
        d = SX.sym('d', dim_d)

        inter_var = np.dot(V, np.linalg.matrix_power(A, self.r))
        cost_vec  = - mtimes(inter_var, d)
        g         = mtimes(V, d)

        alpha_vec = [0] * dim_cost_alpha
        for row_idx in range(dim_cost_alpha):
            nlp    = {'x':d, 'f':cost_vec[row_idx], 'g':g}
            solver = nlpsol('solver', 'ipopt', nlp, opts)
            res    = solver(x0=[0.0]*dim_d, ubg=1)

            alpha_vec[row_idx] = float(- res['f'])

        alpha = max(alpha_vec)

        return alpha

    def mRPI_e(self):

        # compute mRPI set S_tilde
        Lv_vertices = np.dot(self.L, self.v_vertices.T).T

        inter_var            = list(it.product(self.w_vertices, Lv_vertices))
        Delta_tilde_vertices = np.array([np.add(w, lv) for w, lv in inter_var])

        hull          = ConvexHull(Delta_tilde_vertices)
        V_Delta_tilde = ((hull.equations[:,:-1].T) / - hull.equations[:,-1]).T

        alpha_e = self.mRPI_polytope(self.A_L, V_Delta_tilde)

        print("e term: alpha: {:.6f}".format(alpha_e))

        Delta_tilde_vertices = polytope_vertices(V_Delta_tilde)

        E_inf_vertices = []

        # find vertices of E_inf
        for pow_idx in range(self.r):
            inter_var = np.dot(np.linalg.matrix_power(self.A_L, pow_idx), Delta_tilde_vertices.T)
            inter_var = (1/(1 - alpha_e)) * inter_var
            E_inf_vertices.append(inter_var.T)

        inter_var      = list(it.product(*E_inf_vertices))
        E_inf_vertices = np.array([sum(tup) for tup in inter_var])

        hull    = ConvexHull(E_inf_vertices)
        V_E_inf = ((hull.equations[:,:-1].T) / - hull.equations[:,-1]).T

        return V_E_inf

    def get_tube_k(self, Z_prior_vertices):
        '''
        get_tube_k returns the tube of z(k+1) = F_dyn z(k) + G_dyn d(k)
        Inputs:     Z_prior_vertices: vertices of z(k)
        Outputs:    V_Z_post: description of z(k+1) in terms of linear inequalities Vz <= 1
                    Z_post_vertices: vertices of z(k+1)
        '''

        FZ_prior_vertices = np.dot(self.F_dyn, Z_prior_vertices.T).T
        GD_vertices       = np.dot(self.G_dyn, self.wv_vertices.T).T

        inter_var       = list(it.product(FZ_prior_vertices, GD_vertices))
        Z_post_vertices = np.array([np.add(fz, gd) for fz, gd in inter_var])

        hull     = ConvexHull(Z_post_vertices)
        V_Z_post = ((hull.equations[:,:-1].T) / - hull.equations[:,-1]).T

        Z_post_vertices = Z_post_vertices[hull.vertices]

        return V_Z_post, Z_post_vertices

    def get_tightened_constraint(self, V_z):

        opts                      = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"]        = 0

        # solve h_vec
        dim_z = np.shape(V_z)[1]
        h_vec = np.zeros(self.dim_h)

        FGK_aug = np.concatenate((self.F, self.F + np.dot(self.G, self.K)), axis=1)

        z        = SX.sym('z', dim_z)
        g        = mtimes(V_z, z)
        cost_vec = - mtimes(FGK_aug, z)

        for row_idx in range(self.dim_h):
            nlp    = {'x':z, 'f':cost_vec[row_idx], 'g':g}
            solver = nlpsol('solver', 'ipopt', nlp, opts)
            res    = solver(x0=[0.0]*dim_z, ubg=1)

            h_vec[row_idx] = float(- res['f'])

        return h_vec

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
        dim_w = len(Q)
        dim_v = len(R)

        # initial constraints and initial guess
        xhat = np.array([[-3.0],[-8.0]])

        # real initial state
        x = np.array([[-3.1],[-8.0]]) # specially designed for this example

        r = 6

        N = 15

        mpc_single_tube = MPC_single_tube(A, B, C, F, G, f, Q, R, K, L, r, N)

        xbar = copy.deepcopy(xhat)

        # visualize the closed-loop trajectory and the control inputs
        xbar_vis = []
        x_vis    = []
        ubar_vis = []
        u_vis    = []

        threshold = 0.2

        # keep iterating until the nominal state converges to 0
        time_step = 0
        while True:

            # compute optimal control
            sol, solver_status = mpc_single_tube.solve(xbar, time_step)

            if solver_status != "Solve_Succeeded":
                print("MPC failed. Control terminated. Error:", solver_status)
                break

            xbar     = np.array(sol["x"][mpc_single_tube.first_state_index.xbar[0]:mpc_single_tube.first_state_index.ubar[0]:mpc_single_tube.horizon])
            ubar_opt = np.array(sol["x"][mpc_single_tube.first_state_index.ubar[0]::(mpc_single_tube.horizon - 1)])
            u_opt = np.dot(K, (xhat - xbar)) + ubar_opt

            xbar_vis.append(xbar.flatten())
            x_vis.append(x.flatten())
            ubar_vis.append(ubar_opt.flatten())
            u_vis.append(u_opt.flatten())

            w = sample_from_ellipsoid(1, np.linalg.inv(Q), np.zeros(dim_w))
            v = sample_from_ellipsoid(1, np.linalg.inv(R), np.zeros(dim_v))

            # estimate the next state
            y    = np.dot(C, x) + v.reshape((dim_v, 1))
            yhat = np.dot(C, xhat)

            xhat = np.dot(A, xhat) + np.dot(B, u_opt) + np.dot(L, yhat - y)

            # simulate forward
            x    = np.dot(A, x) + np.dot(B, u_opt) + w.reshape((dim_w, 1))
            xbar = np.dot(A, xbar) + np.dot(B, ubar_opt)

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

    pickle.dump([traj_random, control_random, traj_nominal, control_nominal],
                 open("results/double_integrator_single_tube.p", "wb"))