# General imports
from typing import List, Optional, Tuple, Union, Dict
import time
from copy import copy

# numpy
import numpy as np
from numpy.linalg import LinAlgError

# Pytorch geometric
from torch_geometric.data import Data

# Pytorch
import torch

from sklearn.preprocessing import normalize

# cvxopt, required to solve the semidefinite program defined by lovasz to compute the lovasz number of a graph
from cvxopt.base import matrix
from cvxopt.base import spmatrix
import cvxopt.solvers as solvers

import scipy.linalg
import scipy.special

# Probably removed later
from itertools import product
import math

import constants

import util

class Lovasz_graph_features():

    def __init__(self, embedding_dim: int):
        super().__init__()

        # Supress print of the iterations when calculating the solution of an SDP
        solvers.options["show_progress"] = False
        
        # Set the maximum number of iterations when calculating the solution of an SDP
        if constants.sdp_max_iters is not None:
            solvers.options["maxiters"] = constants.sdp_max_iters

        assert embedding_dim > 0
        self.embedding_dim = embedding_dim

        self.tolerance = constants.welzl_tolerance

        if constants.random_generator is not None:
            self.rng = constants.random_generator
        else:
            self.rng = np.random.default_rng() # unpredictable, unseeded

        self.P = None # used in welzls move to fron variant

    # graph_size_range represents a tuple of the minimum subset size and maximum subset size that is considered (the values for d, thus the coordinates in the feature vector). 
    # This allows the feature dimension to be more reasonably controlled, especially if the graph sizes are very uneven.
    # num_samples is the number of random samples drawn for a given dimension (s_d in the original paper)
    def compute_lovasz_feature_vector(self, vertex_identifier: Tuple[int,int], graph_size_range: Tuple[int, int], orthonormal_representation: np.ndarray, num_samples: int) -> Tuple[np.ndarray, float]:
        
        t0 = time.time()

        min_size, max_size = graph_size_range
        num_vertices = orthonormal_representation.shape[1]

        result = np.zeros(shape = (2 + (max_size - min_size) + 1,), dtype = np.float64)
        result[0] = vertex_identifier[0]
        result[1] = vertex_identifier[1]

        sample_distribution = self.get_sample_distribution(graph_sizes_range = graph_size_range, num_vertices = num_vertices, num_samples = num_samples)

        for d, cur_num_samples in sample_distribution.items():
            # Randomly sample subsets
            lovasz_values = []

            found_indices = np.zeros(shape = (cur_num_samples, d), dtype = int) # Store the already found indices

            for idx in range(cur_num_samples):
                if d == num_vertices:
                    indices = np.asarray(range(num_vertices))
                else:
                    # Generate cur_num_samples unique samples of cardinality d
                    is_new = False
                    while not is_new:
                        indices = self.rng.choice(a = num_vertices, size = d, replace = False)
                        if not np.any(np.all(indices == found_indices, axis = 0)):
                            found_indices[idx, :] = indices
                            is_new = True
                    
                # indices is the newly generated subset
                lovasz_values.append(self.compute_min_enclosing_cone_theta(U = orthonormal_representation[:,indices]))

            d_idx = 2 + (d - min_size) - 1

            result[d_idx] = np.mean(lovasz_values)

        return result, time.time() - t0


    # Returns the number of samples per d in the range for a given graph. Implementation analogous to the GraKel implementation
    def get_sample_distribution(self, graph_sizes_range: Tuple[int,int], num_vertices: int, num_samples: int) -> Dict[int, int]:
        min_graph_size, max_graph_size = graph_sizes_range
        
        max_dimension = min(max_graph_size, num_vertices) # We cannot consider samples that include vertices that do not exist in the given graph

        # sample ratios is the ratio of subsets with the given number of vertices (k) of the overall number of possible subsets
        sample_ratios = np.array([scipy.special.binom(num_vertices, k) for k in range(min_graph_size, max_dimension + 1)], dtype = float)
        sample_ratios = sample_ratios / np.sum(sample_ratios)

        samples = np.floor(sample_ratios * num_samples).astype(int)
        samples_size = samples.shape[0]

        for idx in range(int(num_samples - np.sum(samples))):
            # The total number of samples might not be sufficiently achieved, we add one for every such sample starting from the largest subset
            samples[(samples_size - idx - 1) % samples_size] += 1

        return { idx + min_graph_size : samples[idx] for idx in range(samples_size) if samples[idx] > 0.0}

    # Implementation oriented from https://github.com/ysig/GraKeL/tree/master
    # NOTE: This implementation only works with undirected graphs WITHOUT self-loops
    # Returns the lovasz theta number and the slack matrix S of the sdp
    def compute_lovasz_number(self, graph: Data, num_vertices: int) -> Tuple[float, np.ndarray]:

        if num_vertices == 1:
            return 1.0
        
        # Supress print of the iterations when calculating the solution of an SDP
        solvers.options["show_progress"] = False
        
        # Set the maximum number of iterations when calculating the solution of an SDP
        if constants.sdp_max_iters is not None:
            solvers.options["maxiters"] = constants.sdp_max_iters
        
        # Solve sdp formulation given by Lovasz in the paper about the Shannon capacity of graphs:
        # Let {1,...,n} be the vertices of the given graph. Let B in R^nxn range over all p.s.d. matrices such that
        # b_ij = 0 if (i,j) in E(G) and Tr(B) = 1
        # Then: theta(G) = max_B (Tr(B J)) with J being an nxn matrix of ones.

        # When defining A.B as sum_ij A_ij B_ij we can formulate this as:
        # Max B.J
        # s. th. (1 to |E|): b_ij = 0 if (i,j) in E <=> B.(0,...,0,1,0,...,0) = 0 with the 1 at index (i,j)
        #         (|E| + 1): sum_i b_ii = 1 <=> B.I_n = 1 with I_n being the nxn identity matrix
        #                    B >= 0, B is pos. semidefinite

        # We formulate the dual to the above problem (according to https://ocw.mit.edu/courses/15-084j-nonlinear-programming-spring-2004/a632b565602fd2eb3be574c537eea095_lec23_semidef_opt.pdf)
        # Min sum_(i=1)^(|E|+1) y_i (0,...,0,1)_i = y_(|E|+1)
        # s. th. sum_(i=1)^(|E|) y_i (0,...,0,1,0,...,0) + y_(|E|+1) I_n + S = J
        #        S >= 0

        # Extract the edges of graph and remove the reverse edges
        edges = []
        if graph.edge_index is None or graph.edge_index[0].shape[0] == 0:
            num_edges = 0
        else:
            row, col = graph.edge_index
            num_edges = row.shape[0]

            for e in range(num_edges):
                i, j = int(row[e].item()), int(col[e].item())
                t, t_rev = tuple([i,j]), tuple([j,i])
                if t_rev not in edges:
                    edges.append(t)

            num_edges = len(edges)

        # Target function
        # sum_(i=1)^(|E|+1) y_i (0,...,0,1)_i = y_(|E|+1) is encoded. y_i are the |E|+1 optimization variables, the target function is c^tr y
        c = matrix([0.0] * num_edges + [1.0])

        # Create empty sparse matrix of shape (num_vertices^2, num_edges+1)
        # Each column of G represents a matrix, each row corresponds to the |E|+1 optimization variables of the dual problem.
        G = spmatrix(0, [], [], (num_vertices * num_vertices, num_edges + 1))

        # For each edge (each column) we create the matrix corresponding to the edge given by (0,...,0,1,0,...,0) with 1 at (i,j) = e
        # The constraints are evaluated in the form sum_(i = 1)^(|E|+1) y_i * G[:,i] + S = J
        for (e, (i,j)) in enumerate(edges):
            G[i * num_vertices + j, e] = 1
            G[j * num_vertices + i, e] = 1

        for i in range(num_vertices):
            G[i * num_vertices + i, num_edges] = 1

        # We need to change the signs of G and h 
        G = -G

        # h is the supposed result of the constraints (here: J)
        h = -matrix(1.0, (num_vertices, num_vertices))

        sol = solvers.sdp(c = c, Gs = [G], hs = [h])

        # The solution of the sdp is the maximized value of the target function, in this case x_(|E|+1)
        theta = sol["x"][num_edges]
        # sol["ss"] refers to the slack variables of the second order inequalities (the only ones set by us (defined by G and h)) of the sdp.
        # note that this is a symmetric positive semidefinite matrix by definition.
        S = np.asarray(sol["ss"][0])

        return theta, S
        
    # embedding_dim should be the maximum graph order + 1. This ensures that the embedding dimension is sufficiently large and that we can easily construct an
    # orthogonal vector by choosing the last unit vector
    def compute_orthonormal_basis(self, theta: float, S: np.array, embedding_dim: int) -> np.ndarray:

        # We utilise Prop 5.1 in https://ocw.mit.edu/courses/15-084j-nonlinear-programming-spring-2004/a632b565602fd2eb3be574c537eea095_lec23_semidef_opt.pdf
        # Meaning assuming there exist a feasible solution of the primal SDP, we can get the optimal solution B via SB = 0 -> 
        
        # S is (theta I_n - A) from the orignial Lovasz number paper
        try:
            X = scipy.linalg.cholesky(S)
        except LinAlgError:
            # We add some small scalar times the identity matrix to S due to potential rounding errors
            # We calculate the smallest eigenvalue of S
            min_eig = np.min(np.linalg.eigvalsh(S))
            if abs(min_eig) < constants.lo_small_val:
                S += 2 * constants.lo_small_val * np.identity(n = S.shape[0])
            else:
                S += 2 * abs(min_eig) * np.identity(n = S.shape[0])
            # We can assume that min_eig is very close to zero due to S being the result of a semidefinite program
            X = scipy.linalg.cholesky(S)

        cur_dim = X.shape[0]

        # x_i is given by the i-th column of X, we need to pad its dimension to be consistent with the embedding dimension. We do not pad at the beginning but the end of the vectors
        X = np.pad(X, [(0, embedding_dim - cur_dim),(0, 0)], mode = 'constant', constant_values = 0)

        # Now we need a vector that is orthogonal to every vector in X. Due to the embedding dimension being at least one bigger than the largest possible dimension
        # of X, this is done by choosing the embedding_dims unit vector.
        c = np.zeros(shape = (embedding_dim,))
        c[-1] = 1

        # the i-th basis vector is given by 1/sqrt(theta) (c + x_i). To compute the whole basis in one step, we construct C which has c as its columns
        C = np.outer(c, np.ones(shape = (cur_dim,)))

        U = (1/math.sqrt(theta)) * (C + X)

        # U = normalize(U, axis = 0)

        return U
    
    # U is an array whose columns represent the vectors that should be enclosed.
    def compute_min_enclosing_cone_theta(self, U: np.ndarray) -> float:
        
        num_vectors = U.shape[1]

        # We use welzls algorithm to compute the center of the disk represented by U, thus compute the handle c of the cone.
        # According to Welzl, using a single permutation of U at the beginning is sufficient to improve performance
        P = self.rng.permutation(num_vectors)
        R = np.asarray([], dtype = int)
        
        # t0 = time.time()
        c, _ = self.welzl_enclosing_ball(U = U.copy(), P = P.copy(), R = R.copy())
        # rec_time = time.time() - t0

        # t0_it = time.time()
        # c_it, _ = self.welzl_enclosing_ball_iterative(U = U.copy(), P = P.copy(), R = R.copy())
        # it_time = time.time() - t0_it
        # print(f"Diff Recursive - iterative: {rec_time - it_time}")

        # self.P = P

        # t0_mtf = time.time()
        # c_mtf, _ = self.welzl_enclosing_ball_move_to_front(U = U.copy(), P_complement = np.asarray([], dtype = int), R = R.copy())
        # mtf_time = time.time() - t0_mtf
        # print(f"Diff recursive - mtf: {rec_time - mtf_time}")
        # print(f"Time mtf: {mtf_time}")
        
        # We need to normalize c according to https://proceedings.mlr.press/v32/johansson14.pdf
        norm_c = np.linalg.norm(c, 2)
        if norm_c != 0:
            c = c / norm_c

        # theta = max_{u_i} 1/(c^tr u_i)^2
        theta = -1.0
        for i in range(num_vectors):
            val = np.dot(c, U[:,i]).item() ** 2
            if val != 0.0:
                val = 1/val
            if val > theta:
                theta = val

        return theta
    
    # Implements welzls algorithm to recursively compute the smallest enclosing disk in a hyperplane of a set of vectors
    # see: https://www.ibr.cs.tu-bs.de/courses/ws2122/ag/otherstuff/smallest-disk-welzl.pdf
    # P is a gives a permutation of indices of vectors of U that are considered. In recursive calls, elements of P are removed.
    # R is a set of indices of vectors that are considered border vectors of the disk.
    def welzl_enclosing_ball(self, U: np.ndarray, P: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:

        embedding_dim = U.shape[0]
        num_p = P.shape[0]
        num_r = R.shape[0]
        
        # Analogous to Welzls algorithm since we generalize from the 2-dim case, we need to test for embedding_dim + 1
        if num_p == 0 or num_r == embedding_dim + 1:
            if num_r == 0:
                return np.zeros(shape = (embedding_dim,)), 0.0
            else:
                # We have d+1 border points and can thus solve for the center
                c, r = self.calc_min_ball(U[:,R])
        else:
            # randomly choose a point p
            p = P[self.rng.integers(low = 0, high = num_p)]
            P_prime = np.delete(P, np.where(P == p))
            # Calculate the minidisk without p
            c, r = self.welzl_enclosing_ball(U, P_prime, R)
            if abs(np.linalg.norm(U[:,p] - c) - r) > self.tolerance:
                # p is not contained in the disk generated without p, thus p is a border point for this choice of P and R
                if p not in R:
                    R_prime = np.pad(R, [(0, 1)], mode='constant', constant_values=p)
                    c, r = self.welzl_enclosing_ball(U, P_prime, R_prime)

        return c, r
    
    # Implements welzls algorithm to recursively compute the smallest enclosing disk in a hyperplane of a set of vectors
    # see: https://www.ibr.cs.tu-bs.de/courses/ws2122/ag/otherstuff/smallest-disk-welzl.pdf
    # P is a gives a permutation of indices of vectors of U that are considered. In recursive calls, elements of P are removed.
    # R is a set of indices of vectors that are considered border vectors of the disk.
    def welzl_enclosing_ball_move_to_front(self, U: np.ndarray, P_complement: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:

        embedding_dim = U.shape[0]
        P = np.delete(self.P, np.argwhere(np.isin(self.P, P_complement)))
        num_p = P.shape[0]
        num_r = R.shape[0]
        
        # Analogous to Welzls algorithm since we generalize from the 2-dim case, we need to test for embedding_dim + 1
        if num_p == 0 or num_r == embedding_dim + 1:
            if num_r == 0:
                return np.zeros(shape = (embedding_dim,)), 0.0
            else:
                # We have d+1 border points and can thus solve for the center
                c, r = self.calc_min_ball(U[:,R])
        else:
            # choose the last point in P since P is already a random permutation of the points
            p = P[-1]
            P_complement_prime = np.pad(P_complement, [(0,1)], mode = 'constant', constant_values = p)
            # Calculate the minidisk without p
            c, r = self.welzl_enclosing_ball_move_to_front(U, P_complement_prime, R)
            if abs(np.linalg.norm(U[:,p] - c) - r) > self.tolerance:
                # p is not contained in the disk generated without p, thus p is a border point for this choice of P and R
                if p not in R:
                    R_prime = np.pad(R, [(0, 1)], mode='constant', constant_values=p)
                    c, r = self.welzl_enclosing_ball_move_to_front(U, P_complement_prime, R_prime)
                    # move p to the first position
                    self.P = np.delete(self.P, np.where(self.P == p))
                    self.P = np.pad(self.P, [(1,0)], mode = 'constant', constant_values = p)
                    

        return c, r
    
    # Iterative version of the recursive implementation using a stack to try optimizing the calculation
    def welzl_enclosing_ball_iterative(self, U: np.ndarray, P: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:

        # Utilises a stack given by a list
        # Each iteration has to pass the parameters P, R
        stack = []
        # The results are Dicts of c, r
        results = []
        # If stage == 1, it is handled as a normal call. If stage == 2 the check whether a given point is in a calculated ball is performed
        # Each call gives the parameters stage, P, R, p where p is only used if stage == 2 to check.
        
        embedding_dim = U.shape[0]

        # initial call of the function
        stack.append(tuple((1, P.copy(), R.copy(), None)))
        
        # loop
        while(len(stack) > 0):
            stage, P, R, p = stack.pop()

            if stage == 1:
                num_p = P.shape[0]
                num_r = R.shape[0]

                if num_p == 0 or num_r == embedding_dim + 1:
                    if num_r == 0:
                        results.append(tuple((np.zeros(shape = (embedding_dim,)), 0.0)))
                    else:
                        # We have d+1 border points and can thus solve for the center
                        c, r = self.calc_min_ball(U[:,R])
                        results.append(tuple((c, r)))

                else:
                    # randomly choose a point p
                    p = P[self.rng.integers(low = 0, high = num_p)]
                    # Calculate the minidisk without p
                    P_prime = np.delete(P, np.where(P == p))

                    # Recursive call in the original

                    # Then check whether the point is in the ball (since we use a stack, we need to append using this reversed order)
                    stack.append(tuple((2, P_prime.copy(), R.copy(), p.copy())))
                    # First compute c and r (the recursive step)
                    stack.append(tuple((1, P_prime.copy(), R.copy(), None)))

            elif stage == 2:
                # c, r = self.welzl_enclosing_ball(U, P_prime, R)
                c, r = results.pop()

                if abs(np.linalg.norm(U[:,p] - c) - r) > self.tolerance:
                    # p is not contained in the disk generated without p, thus p is a border point for this choice of P and R
                    if p not in R:
                        R_prime = np.pad(R, [(0, 1)], mode='constant', constant_values=p)
                        # Recursive call
                        stack.append(tuple((1, P.copy(), R_prime.copy(), None)))
                else:
                    results.append(c, r)

        return results.pop()

    # Only temporary
    def calc_min_ball(self, A: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit the minimum ball.

        Parameters
        ----------
        A : np.array, ndim=2
            The vectors that will be enclosed inside the ball.

        Returns
        -------
        c : np.array, ndim=1
            The center vector C of the ball.

        r : int
            The ball radius.

        """
        d = A.shape[0]
        n = A.shape[1]

        if n == 1:
            c, r = A[:, 0], 0
        else:
            Q = A-np.outer(A[:, 0], np.ones(shape=(n, 1)))
            B = 2*np.dot(Q.T, Q)
            b = B.diagonal()/2

            L = np.linalg.solve(B[1:, :][:, 1:], b[1:])
            L = np.pad(L, [(1, 0)], mode='constant', constant_values=0)

            C = np.zeros(shape=(d,))

            for i in range(1, n):
                C = C + L[i]*Q[:, i]

            r = np.sqrt(np.dot(C, C))
            c = C + A[:, 0]

        return c, r

    # # Computes the center and radius of the minimal ball conaining all points in A. It is assumed that A only contains border points
    # def calc_min_ball(self, A: np.ndarray) -> Tuple[np.ndarray, float]:

    #     num_vectors = A.shape[1]
    #     if num_vectors == 1:
    #         return A[:,0], 0.0
    #     elif num_vectors < A.shape[1] + 1:
    #         # Not enough points are given to compute the center in the original dimensions, thus we reduce the dimensionality of the vectors
    #         pass

    #     # The center of the circumhypersphere defined by A is given by the intersection of the hyperplanes bisecting any two points.
    #     # It is sufficient to consider the hyperplanes bisecting the first point to each other point.
    #     # First, we compute these bisecting hyperplanes by computing their normal vectors and a point on the hyperplane (via taking the middle point between two vertices)
    #     normals = A[:,1:] - A[:,0].reshape((-1,1))

    #     # The surface points of those hyperplanes are given by simply taking the middle point between each pair of points
    #     surface_points = (A[:,1:] + A[:,0].reshape((-1,1)))/2

    #     # Now the i-th hyperplane is defined by surface_points[:,i] and normals[:,i].
    #     # x lies on the plane iff normals[:,i] * (surface_points[:,i] - x) = 0
    #     # We can describe the hyperplane as normals[:,i] * x = p for every x on the plane and a constant p (as an unnormalized Hessian Normal form). 
    #     # We compute this p using the surface point
    #     plane_p = np.diagonal(np.dot(np.transpose(normals), surface_points))

    #     # Compute the intersection of the planes to compute the center point
    #     center = np.linalg.solve(np.transpose(normals), plane_p)

    #     # compute the radius (any point can be used)
    #     radius = np.linalg.norm(A[:,0] - center, 2)

    #     return center, radius


# Only for testing
def _fitball_(A):
    """Fit the minimum ball.

    Parameters
    ----------
    A : np.array, ndim=2
        The vectors that will be enclosed inside the ball.

    Returns
    -------
    c : np.array, ndim=1
        The center vector C of the ball.

    r : int
        The ball radius.

    """
    d = A.shape[0]
    n = A.shape[1]

    if n == 1:
        c, r = A[:, 0], 0
    else:
        Q = A-np.outer(A[:, 0], np.ones(shape=(n, 1)))
        B = 2*np.dot(Q.T, Q)
        b = B.diagonal()/2

        L = np.linalg.solve(B[1:, :][:, 1:], b[1:])
        L = np.pad(L, [(1, 0)], mode='constant', constant_values=0)

        C = np.zeros(shape=(d,))

        for i in range(1, n):
            C = C + L[i]*Q[:, i]

        r = np.sqrt(np.dot(C, C))
        c = C + A[:, 0]

    return c, r

# Testing
def is_pos_semidefinite(m: np.array) -> bool:
    return np.all(np.linalg.eigvals(m) >= 0)

if __name__ == '__main__':

    util.initialize_random_seeds(constants.SEED)

    # Test the fitball method
    # d = 4
    p1 = np.asarray([1,2,3,4]).reshape((-1,1))
    p2 = np.asarray([1,3,3,4]).reshape((-1,1))
    p3 = np.asarray([0,0,1,1]).reshape((-1,1))
    p4 = np.asarray([0,0,0,0]).reshape((-1,1))
    p5 = np.asarray([4,4,4,4]).reshape((-1,1))

    points = [p1,p2,p5]

    A = np.concatenate(tuple(points), axis = 1)
    n = A.shape[1]

    lf = Lovasz_graph_features(embedding_dim = 4)
    #c, r = lf.welzl_enclosing_ball(U = A, P = np.random.permutation(n) - 1, R = np.asarray([], dtype = int))
    try:
        c, r = lf.calc_min_ball(A)
    except Exception as e:
        c = np.zeros(shape = (A.shape[0],))
        r = 0.0
        print(repr(e))

    # verify
    c_grakel, r_grakel = _fitball_(A)

    print('finished test')
    print(f'center diff: {c - c_grakel}')
    print(f'radius diff: {r - r_grakel}')
    for idx, point in enumerate(points):
        point = point.reshape(-1)
        #diff = np.linalg.norm(point - c, 2)
        diff = np.sqrt(np.dot(point-c, point-c))
        print(f'Point {idx} distance from center: {diff}, radius: {r}, correct: {np.allclose(np.asarray([diff]), r)}')
        #diff_grakel = np.linalg.norm(point - c_grakel, 2)
        diff_grakel = np.sqrt(np.dot(point-c_grakel, point-c_grakel))
        print(f'Point {idx} distance from GraKel center: {diff_grakel}, radius: : {r_grakel}, correct: {np.allclose(np.asarray([diff_grakel]), r_grakel)}')

    # n = 4

    # # Testing for some graphs with known values (Graphs and values taken from the Wikipedia page of the Lovasz number)

    # # Complete graph
    # n_complete = n

    # edge_index_complete = torch.zeros([2, n_complete**2 - n_complete])
    # e = 0
    # for (i,j) in product(range(n_complete), range(n_complete)):
    #     if i == j:
    #         continue
    #     edge_index_complete[:,e] = torch.tensor([i,j])
    #     e += 1
    # x = torch.zeros(n_complete)
    # graph_complete = Data(x = x, edge_index = edge_index_complete)

    # # Empty graph
    # n_empty = n
    # graph_empty = Data(x = torch.zeros(n_empty))

    # # Pentagon graph
    # n_pent = 5
    # edge_index_pent = torch.zeros([2, 10])
    # edge_index_pent[:,0] = torch.tensor([0,1])
    # edge_index_pent[:,1] = torch.tensor([1,0])
    # edge_index_pent[:,2] = torch.tensor([1,2])
    # edge_index_pent[:,3] = torch.tensor([2,1])
    # edge_index_pent[:,4] = torch.tensor([2,3])
    # edge_index_pent[:,5] = torch.tensor([3,2])
    # edge_index_pent[:,6] = torch.tensor([3,4])
    # edge_index_pent[:,7] = torch.tensor([4,3])
    # edge_index_pent[:,8] = torch.tensor([4,0])
    # edge_index_pent[:,9] = torch.tensor([0,4])
    # graph_pent = Data(x = torch.zeros(n_pent), edge_index = edge_index_pent)

    # lf = Lovasz_graph_features(100000000)

    # theta_complete, S_complete = lf.compute_lovasz_number(graph = graph_complete, num_vertices = n_complete)
    # print(f"theta(K_{n_complete}) = {theta_complete}; Should be {1}.")
    # print(f"Slack vairables: \n{S_complete}")
    # print(f"Is pos semidefinite: {is_pos_semidefinite(S_complete)}")

    # U = lf.compute_orthonormal_basis(theta = theta_complete, S = S_complete, embedding_dim = n_complete + 1)
    # theta_complete_cone = lf.compute_min_enclosing_cone_theta(U = U)
    # print(f"Theta from orthonormal basis: {theta_complete_cone}")

    # print("")

    # theta_empty, S_empty = lf.compute_lovasz_number(graph = graph_empty, num_vertices = n_empty)
    # print(f"theta(Complement of K_{n_empty}) = {theta_empty}; Should be {n_empty}.")
    # print(f"Slack vairables: \n{S_empty}")
    # print(f"Is pos semidefinite: {is_pos_semidefinite(S_empty)}")

    # U = lf.compute_orthonormal_basis(theta = theta_empty, S = S_empty, embedding_dim = n_empty + 1)
    # theta_empty_cone = lf.compute_min_enclosing_cone_theta(U = U)
    # print(f"Theta from orthonormal basis: {theta_empty_cone}")

    # print("")

    # theta_pent, S_pent = lf.compute_lovasz_number(graph = graph_pent, num_vertices = n_pent)
    # print(f"theta(C_5) = {theta_pent}; Should be {math.sqrt(5)}.")
    # print(f"Slack vairables: \n{S_pent}")
    # print(f"Is pos semidefinite: {is_pos_semidefinite(S_pent)}")

    # U = lf.compute_orthonormal_basis(theta = theta_pent, S = S_pent, embedding_dim = n_pent + 1)
    # theta_pent_cone = lf.compute_min_enclosing_cone_theta(U = U)
    # print(f"Theta from orthonormal basis: {theta_pent_cone}")
