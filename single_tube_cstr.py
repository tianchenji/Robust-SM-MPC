# ---------------------------------------------------------------------------
# Robust Output Feedback MPC -- Single Tube Approach
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2022-02-24
# Paper: M. Kogel and R. Findeisen. "Robust output feedback MPC for uncertain 
# linear systems with reduced conservatism." IFAC-PapersOnLine, 2017.
# ---------------------------------------------------------------------------

import sys
import pickle
import itertools as it
import matplotlib.pyplot as plt

from casadi import *
from math import sqrt
from scipy.spatial import ConvexHull

from utils import polytope_vertices

class MPC_single_tube:

    def __init__(self, A, B, C, F, G, f, K, L, r):

        self.A, self.B, self.C = A, B, C
        self.F, self.G, self.f = F, G, f.flatten()
        self.K, self.L         = K, L
        self.r                 = r

        self.A_L = self.A + np.dot(self.L, self.C)

        F_dyn_row_1 = np.concatenate((self.A + np.dot(self.L, self.C), np.zeros_like(self.A)), axis=1)
        F_dyn_row_2 = np.concatenate((- np.dot(self.L, self.C), self.A + np.dot(self.B, self.K)), axis=1)
        self.F_dyn  = np.concatenate((F_dyn_row_1, F_dyn_row_2), axis=0)

        G_dyn_row_1 = np.concatenate((np.identity(len(self.L)), self.L), axis=1)
        G_dyn_row_2 = np.concatenate((np.zeros_like(self.A), - self.L), axis=1)
        self.G_dyn  = np.concatenate((G_dyn_row_1, G_dyn_row_2), axis=0)

        # constraints on state and output disturbances
        self.w_vertices = np.array([[-0.2, -0.2],
                                    [-0.2, 0.2],
                                    [0.2, -0.2],
                                    [0.2, 0.2]])
        self.v_vertices = np.array([[-0.1], [0.1]])

        self.wv_vertices = list(it.product(self.w_vertices, self.v_vertices))
        self.wv_vertices = np.array([np.concatenate((w, v), axis=0) for w, v in self.wv_vertices])

    def mRPI_polytope(self, A, V):
        '''
        mRPI_polytope computes the minimal RPI set of a given dynamics e = Ae + d
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

        Lv_vertices = np.dot(self.L, self.v_vertices.T).T

        inter_var            = list(it.product(self.w_vertices, Lv_vertices))
        Delta_tilde_vertices = np.array([np.add(w, lv) for w, lv in inter_var])

        hull          = ConvexHull(Delta_tilde_vertices)
        V_Delta_tilde = ((hull.equations[:,:-1].T) / - hull.equations[:,-1]).T

        # compute mRPI set S_tilde
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
        get_tube_k returns the tube of z(k+1) = Fz(k) + Gd(k)
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
        dim_z      = np.shape(V_z)[1]
        dim_cost_h = len(self.F)
        h_vec      = np.zeros(dim_cost_h)

        FGK_aug = np.concatenate((self.F, self.F + np.dot(self.G, self.K)), axis=1)

        z        = SX.sym('z', dim_z)
        g        = mtimes(V_z, z)
        cost_vec = - mtimes(FGK_aug, z)

        for row_idx in range(dim_cost_h):
            nlp    = {'x':z, 'f':cost_vec[row_idx], 'g':g}
            solver = nlpsol('solver', 'ipopt', nlp, opts)
            res    = solver(x0=[0.0]*dim_z, ubg=1)

            h_vec[row_idx] = float(- res['f'])

        return h_vec


if __name__ == '__main__':

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

    r = 6

    mpc_single_tube = MPC_single_tube(A, B, C, F, G, f, K, L, r)

    tightened_constraint = []

    # compute Z_0 in terms of linear inequalities
    V_E_inf     = mpc_single_tube.mRPI_e()
    V_E_inf_aug = np.concatenate((V_E_inf, np.zeros_like(V_E_inf)), axis=1)

    V_Xi     = np.concatenate((1e5*np.identity(len(A)), -1e5*np.identity(len(A))), axis=0)
    V_Xi_aug = np.concatenate((np.zeros_like(V_Xi), V_Xi), axis=1)

    V_Z_prior        = np.concatenate((V_E_inf_aug, V_Xi_aug), axis=0)
    Z_prior_vertices = polytope_vertices(V_Z_prior)

    tightened_constraint.append(mpc_single_tube.get_tightened_constraint(V_Z_prior))

    for i in range(10):
        V_Z_post, Z_post_vertices = mpc_single_tube.get_tube_k(Z_prior_vertices)
        Z_prior_vertices = Z_post_vertices

        tightened_constraint.append(mpc_single_tube.get_tightened_constraint(V_Z_post))

    '''
    # compute (I I)Z
    X_tightened_vertices = Z_post_vertices[:,:2] + Z_post_vertices[:,2:]

    hull = ConvexHull(X_tightened_vertices)

    plt.fill(X_tightened_vertices[hull.vertices, 0], X_tightened_vertices[hull.vertices, 1], color='green', alpha=0.5)
    '''

    tightened_constraint = np.array(tightened_constraint)
    
    pickle.dump(tightened_constraint, open("results/cstr_tightening_single_tube.p", "wb"))

    plt.plot(tightened_constraint[:,2], marker='.')
    plt.show()