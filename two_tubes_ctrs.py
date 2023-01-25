# ---------------------------------------------------------------------------
# Robust Output Feedback MPC -- Two Tubes Approach
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2022-03-05
# Paper: D.Q. Mayne, et al. "Robust output feedback model predictive control
# of constrained linear systems." Automatica, 2006.
# ---------------------------------------------------------------------------

import sys
import pickle
import itertools as it
import matplotlib.pyplot as plt

from casadi import *
from math import sqrt
from scipy.spatial import ConvexHull

from utils import polytope_vertices

class MPC_two_tubes:

    def __init__(self, A, B, C, F, G, f, K, L, r):

        self.A, self.B, self.C = A, B, C
        self.F, self.G, self.f = F, G, f.flatten()
        self.K, self.L         = K, L
        self.r                 = r

        self.A_L = self.A + np.dot(self.L, self.C)
        self.A_K = self.A + np.dot(self.B, self.K)

        # constraints on state and output disturbances
        self.w_vertices = np.array([[-0.2, -0.2],
                                    [-0.2, 0.2],
                                    [0.2, -0.2],
                                    [0.2, 0.2]])
        self.v_vertices = np.array([[-0.1], [0.1]])

        self.V_E_inf, E_inf_vertices = self.mRPI_e()
        Lv_vertices  = - np.dot(self.L, self.v_vertices.T).T
        LCe_vertices = - np.dot(self.L, np.dot(self.C, E_inf_vertices.T)).T

        inter_var          = list(it.product(LCe_vertices, Lv_vertices))
        self.Delta_bar_vertices = np.array([np.add(lce, lv) for lce, lv in inter_var])

        #hull = ConvexHull(Delta_bar_vertices)
        #self.Delta_bar_vertices = Delta_bar_vertices[hull.vertices]

        self.h_e_vec = self.get_tightened_constraint_e()

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

        return V_E_inf, E_inf_vertices[hull.vertices]

    def get_tightened_constraint_e(self):

        opts                      = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"]        = 0

        # solve h_e_vec
        dim_e        = len(self.A)
        dim_cost_h_e = len(self.F)
        h_e_vec      = np.zeros(dim_cost_h_e)

        e        = SX.sym('e', dim_e)
        g        = mtimes(self.V_E_inf, e)
        cost_vec = - mtimes(self.F, e)

        for row_idx in range(dim_cost_h_e):
            nlp    = {'x':e, 'f':cost_vec[row_idx], 'g':g}
            solver = nlpsol('solver', 'ipopt', nlp, opts)
            res    = solver(x0=[0.0]*dim_e, ubg=1)

            h_e_vec[row_idx] = float(- res['f'])

        return h_e_vec

    def get_tube_xi_k(self, Xi_prior_vertices):
        '''
        get_tube_k returns the tube of xi(k+1) = (A + BK)xi(k) + delta_bar(k)
        Inputs:     Xi_prior_vertices: vertices of xi(k)
        Outputs:    V_Xi_post: description of xi(k+1) in terms of linear inequalities Vxi <= 1
                    Xi_post_vertices: vertices of xi(k+1)
        '''

        A_KXi_prior_vertices = np.dot(self.A_K, Xi_prior_vertices.T).T

        inter_var       = list(it.product(A_KXi_prior_vertices, self.Delta_bar_vertices))
        Xi_post_vertices = np.array([np.add(a_kxi, delta_bar) for a_kxi, delta_bar in inter_var])

        hull      = ConvexHull(Xi_post_vertices)
        V_Xi_post = ((hull.equations[:,:-1].T) / - hull.equations[:,-1]).T

        Xi_post_vertices = Xi_post_vertices[hull.vertices]

        return V_Xi_post, Xi_post_vertices

    def get_tightened_constraint_xi(self, V_xi):

        opts                      = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"]        = 0

        # solve h_xi_vec
        dim_xi        = len(self.A)
        dim_cost_h_xi = len(self.F)
        h_xi_vec      = np.zeros(dim_cost_h_xi)

        FGK = self.F + np.dot(self.G, self.K)

        xi       = SX.sym('xi', dim_xi)
        g        = mtimes(V_xi, xi)
        cost_vec = - mtimes(FGK, xi)

        for row_idx in range(dim_cost_h_xi):
            nlp    = {'x':xi, 'f':cost_vec[row_idx], 'g':g}
            solver = nlpsol('solver', 'ipopt', nlp, opts)
            res    = solver(x0=[0.0]*dim_xi, ubg=1)

            h_xi_vec[row_idx] = float(- res['f'])

        return h_xi_vec


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

    mpc_two_tubes = MPC_two_tubes(A, B, C, F, G, f, K, L, r)

    tightened_constraint = []

    V_Xi = np.concatenate((1e5*np.identity(len(A)), -1e5*np.identity(len(A))), axis=0)
    Xi_prior_vertices = polytope_vertices(V_Xi)

    h_e_vec = mpc_two_tubes.h_e_vec
    tightened_constraint.append(h_e_vec)

    for i in range(10):
        V_Xi_post, Xi_post_vertices = mpc_two_tubes.get_tube_xi_k(Xi_prior_vertices)
        Xi_prior_vertices = Xi_post_vertices

        h_xi_vec = mpc_two_tubes.get_tightened_constraint_xi(V_Xi_post)
        tightened_constraint.append(h_e_vec + h_xi_vec)

    tightened_constraint = np.array(tightened_constraint)

    pickle.dump(tightened_constraint, open("results/cstr_tightening_two_tubes.p", "wb"))

    plt.plot(tightened_constraint[:,4], marker='.')
    plt.show()