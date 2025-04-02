# General imports
from typing import List, Optional, Tuple, Union, Dict

# numpy
import numpy as np

# Pytorch geometric
from torch_geometric.data import Data

# Pytorch
import torch

# cvxopt, required to solve the semidefinite program defined by lovasz to compute the lovasz number of a graph
from cvxopt.base import matrix
from cvxopt.base import spmatrix
import cvxopt.solvers as solvers

# Probably removed later
from itertools import product
import math

class Lovasz_graph_features():

    def __init__(self, dim_max: int, max_iters: Optional[int] = None):
        super().__init__()

        # Supress print of the iterations when calculating the solution of an SDP
        solvers.options["show_progress"] = False
        
        # Set the maximum number of iterations when calculating the solution of an SDP
        if max_iters is not None:
            solvers.options["maxiters"] = max_iters

        assert dim_max > 0
        self.dim_max = dim_max

    # Implementation oriented from https://gist.github.com/dstahlke/6895643
    # NOTE: This implementation only works with undirected graphs WITHOUT self-loops
    # Returns the lovasz theta number and the slack variable S of the dual sdp
    def compute_lovasz_number(self, graph: Data, num_vertices: int) -> Tuple[float, np.array]:

        if num_vertices == 1:
            return 1.0
        
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
        
    def compute_orthonormal_basis(self, theta: float, S: np.array):

        # We utilise Prop 5.1 in https://ocw.mit.edu/courses/15-084j-nonlinear-programming-spring-2004/a632b565602fd2eb3be574c537eea095_lec23_semidef_opt.pdf
        # Meaning assuming there exist a feasible solution of the primal SDP, we can get the optimal solution B via SB = 0 -> 
        
        # S = L L^tr after the cholesky decomposition, L is a lower triangular matrix
        L = np.linalg.cholesky(S)

        raise NotImplementedError

# Testing
def is_pos_semidefinite(m: np.array) -> bool:
    return np.all(np.linalg.eigvals(m) >= 0)

if __name__ == '__main__':
    n = 4

    # Testing for some graphs with known values (Graphs and values taken from the Wikipedia page of the Lovasz number)

    # Complete graph
    n_complete = n

    edge_index_complete = torch.zeros([2, n_complete**2 - n_complete])
    e = 0
    for (i,j) in product(range(n_complete), range(n_complete)):
        if i == j:
            continue
        edge_index_complete[:,e] = torch.tensor([i,j])
        e += 1
    x = torch.zeros(n_complete)
    graph_complete = Data(x = x, edge_index = edge_index_complete)

    # Empty graph
    n_empty = n
    graph_empty = Data(x = torch.zeros(n_empty))

    # Pentagon graph
    n_pent = 5
    edge_index_pent = torch.zeros([2, 10])
    edge_index_pent[:,0] = torch.tensor([0,1])
    edge_index_pent[:,1] = torch.tensor([1,0])
    edge_index_pent[:,2] = torch.tensor([1,2])
    edge_index_pent[:,3] = torch.tensor([2,1])
    edge_index_pent[:,4] = torch.tensor([2,3])
    edge_index_pent[:,5] = torch.tensor([3,2])
    edge_index_pent[:,6] = torch.tensor([3,4])
    edge_index_pent[:,7] = torch.tensor([4,3])
    edge_index_pent[:,8] = torch.tensor([4,0])
    edge_index_pent[:,9] = torch.tensor([0,4])
    graph_pent = Data(x = torch.zeros(n_pent), edge_index = edge_index_pent)

    lf = Lovasz_graph_features()

    theta_complete, S_complete = lf.compute_lovasz_number(graph = graph_complete, num_vertices = n_complete)
    print(f"theta(K_{n_complete}) = {theta_complete}; Should be {1}.")
    print(f"Slack vairables: \n{S_complete}")
    print(f"Is pos semidefinite: {is_pos_semidefinite(S_complete)}")

    print("")

    theta_empty, S_empty = lf.compute_lovasz_number(graph = graph_empty, num_vertices = n_empty)
    print(f"theta(Complement of K_{n_empty}) = {theta_empty}; Should be {n_empty}.")
    print(f"Slack vairables: \n{S_empty}")
    print(f"Is pos semidefinite: {is_pos_semidefinite(S_empty)}")

    print("")

    theta_pent, S_pent = lf.compute_lovasz_number(graph = graph_pent, num_vertices = n_pent)
    print(f"theta(C_5) = {theta_pent}; Should be {math.sqrt(5)}.")
    print(f"Slack vairables: \n{S_pent}")
    print(f"Is pos semidefinite: {is_pos_semidefinite(S_pent)}")
