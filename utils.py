import matplotlib.pyplot as plt

from casadi import *
from math import sqrt
from scipy.spatial import HalfspaceIntersection

class FirstStateIndex:

    def __init__(self, A, B, N):
        '''
        FirstStateIndex serves for readability
        Note: the variable "horizon" includes the current time step
        Inputs:     A, B: system dynamics
                    N: the prediction horizon
        '''

        state_dim = np.shape(A)[1]
        input_dim = np.shape(B)[1]
        horizon   = N + 1

        self.xbar = [0] * state_dim
        self.ubar = [0] * input_dim
        self.xbar[0] = 0
        self.ubar[0] = state_dim * horizon
        for i in range(state_dim - 1):
            self.xbar[i + 1] = self.xbar[i] + horizon
        for i in range(input_dim - 1):
            self.ubar[i + 1] = self.ubar[i] + horizon - 1

def e_project(Shape, center, delta=0):
    '''
    e_project returns bounds of orthogonal projection of ellipsoids to axes
    Inputs:     Shape: the shape matrix of the ellipsoid
                center: the center of the ellipsoid
                delta: the degree of shrinkage
    Outputs:    s_min_i: the lowerbound of the ith state
                s_max_i: the upperbound of the ith state
    '''

    # cholesky decomposition of shape matrix
    L = np.linalg.cholesky(Shape)

    # the center of ellipsoid
    c           = center
    ellipsoid_d = len(center)
    s0          = [None] * ellipsoid_d
    s_min       = [None] * ellipsoid_d
    s_max       = [None] * ellipsoid_d

    for i in range(ellipsoid_d):
        v    = np.zeros(ellipsoid_d)
        v[i] = 1.0

        s0[i]    = np.dot(v, c) / np.dot(v, v)
        w        = np.dot(np.linalg.inv(L), v) / np.dot(v, v)
        norm_w   = np.linalg.norm(w) * sqrt(1 - delta)
        s_min[i] = float(s0[i] - norm_w)
        s_max[i] = float(s0[i] + norm_w)

    return (s_min, s_max)

def sample_from_ellipsoid(num_of_samples, Shape, center, delta=0):
    '''
    sample_from_ellipsoid draw uniform samples from a given ellipsoid
    Inputs:     num_of_samples: the desired number of samples
                Shape: the shape matrix of the ellipsoid
                center: the center of the ellipsoid
                delta: the degree of shrinkage
    Outputs:    samples_inside: uniform samples drawn from the ellipsoid
    '''

    c           = center
    ellipsoid_d = len(center)

    s_min, s_max = e_project(Shape, center, delta)

    sample_batch   = num_of_samples*2
    samples_inside = []

    while len(samples_inside) < num_of_samples:
        samples = np.zeros((ellipsoid_d, sample_batch))
        for i in range(ellipsoid_d):
            samples[i] = np.random.uniform(s_min[i], s_max[i], size=sample_batch)

        samples = samples.T

        for sample in samples:
            if np.dot((sample - c), np.dot(Shape, (sample - c))) <= 1 - delta:
                samples_inside.append(sample)
            if len(samples_inside) == num_of_samples:
                break

    return np.array(samples_inside)

def bounding_box(edges):
    '''
    bounding_box returns the bounding box defined by the extreme values in each coordinate
    Inputs:     edges: the vector [v_max, v_min] containing the extreme values in each coordinate
    Outputs:    V: the matrix describing the bounding box Vx <= 1
    '''

    dim = len(edges) // 2

    max_edges = np.diag(1/edges[:dim])
    min_edges = np.diag(1/edges[dim:])

    V = np.concatenate((max_edges, min_edges))

    return V

def bounding_box_ellipsoid(T, Shape):
    '''
    bounding_box_ellipsoid returns the bounding box of the convex set formed by applying the 
    linear transformation T to the original ellipsoid
    Assumptions:    1. The original ellipse is centered at the origin
    Inputs:     T: the linear transformation matrix
                Shape: the shape matrix of the original ellipsoid
    Outputs:    V: the description of the bounding box in terms of linear inequalities
    '''

    opts                      = {}
    opts["ipopt.print_level"] = 0
    opts["print_time"]        = 0

    epsilon = 1e-5

    dim_x    = len(Shape)
    dim_cost = len(T)

    # define optimization variables
    x = SX.sym('x', dim_x)

    cost_vec  = - mtimes(T, x)
    g         = mtimes([x.T, Shape, x])

    edge_vec = np.zeros(dim_x * 2)
    for row_idx in range(dim_cost):
        nlp    = {'x':x, 'f':cost_vec[row_idx], 'g':g}
        solver = nlpsol('solver', 'ipopt', nlp, opts)
        res    = solver(x0=[0.1]*dim_x, ubg=1)

        edge_vec[row_idx] = float(- res['f']) if float(- res['f']) != 0 else epsilon
        edge_vec[row_idx + dim_cost] = float(res['f']) if float(res['f']) != 0 else - epsilon

    V = bounding_box(edge_vec)

    return V

def polytope_vertices(V):
    '''
    polytope_vertices returns the vertices of the polytope
    Assumptions:    1. The polytope has the origin in its interior
    Inputs:     V: the description of the polytope in terms of linear inequalities Vx <= 1
    Outputs:    vertices
    '''

    dim_x = np.shape(V)[1]

    b   = - np.ones((len(V), 1))
    V_f = np.concatenate((V, b), axis=1)
    halfspace = HalfspaceIntersection(V_f, np.zeros(dim_x))

    return halfspace.intersections

def outbounding_ellipse(T, Shape):
    '''
    outbounding_ellipse returns the outbounding ellipse of the convex set formed by applying the rank deficient 
    linear transformation T to the original ellipse (in 2-D space)
    Assumptions:    1. The original ellipse is centered at the origin
                    2. The linear transformation T is rank deficient
    Inputs:     T: the rank deficient linear transformation
                Shape: the shape matrix of the original ellipsoid
    Outputs:    the shape matrix of the outbounding ellipsoid
    '''

    opts                      = {}
    opts["ipopt.print_level"] = 0
    opts["print_time"]        = 0

    # overage is a hyperparameter
    overage = 5e-2

    x         = SX.sym('x', 2)
    cost      = - mtimes(T, x)
    g         = mtimes([x.T, Shape, x])
    opt_value = [None] * 2

    for dim in range(2):
        nlp    = {'x':x, 'f':cost[dim], 'g':g}
        solver = nlpsol('solver', 'ipopt', nlp, opts)
        res    = solver(ubg=1)

        opt_value[dim] = - res['f']

    opt_value = list(map(float, opt_value))

    major_axis_angle = atan2(opt_value[1], opt_value[0])

    angles = [major_axis_angle, major_axis_angle + pi/2]
    U      = np.array([[cos(angles[0]), cos(angles[1])], [sin(angles[0]), sin(angles[1])]])

    major_semi_axis = float(norm_2(opt_value))
    minor_semi_axis = overage

    Sigma = np.diag([1/major_semi_axis, 1/minor_semi_axis])

    return np.dot(U, np.dot(Sigma, np.dot(Sigma, U.T)))


if __name__ == '__main__':

    T     = np.array([[1.0, 1.0], [0.0, 0.0]])
    Shape = np.array([[1.0, 0.0], [0.0, 1.0]])

    outbounding_shape = outbounding_ellipse(T, Shape)

    samples = sample_from_ellipsoid(4096, outbounding_shape, np.array([0, 0]))

    for sample in samples:
        plt.scatter(sample[0], sample[1], color='b', marker='.')

    plt.grid()
    plt.axis('equal')
    plt.show()