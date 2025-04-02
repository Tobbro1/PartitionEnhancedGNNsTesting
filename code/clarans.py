# Implementation of the CLARANS clustering algorithm (see https://www.researchgate.net/profile/Raymond-Ng-9/publication/3297085_CLARANS_A_method_for_clustering_objects_for_spatial_data_mining/links/0046353398a92d7585000000/CLARANS-A-method-for-clustering-objects-for-spatial-data-mining.pdf).
# We do not use the pyclustering implementation since it is no longer maintained.

# General imports
from typing import Optional, Tuple
import tqdm

# Numpy
import numpy as np

# own imports
import constants

# Clarans is a medoid based clustering algorithm. Note that Clarans is randomized and requires the whole dataset to fit in memory.
# Current issue: in a dataset of 1000000 samples, we need about 1 min to scan each sample ones (for the calculate cost diff method) -> too slow
class Clarans():
    
    # rng_salt should be utilised if multiple clarans instances are generated. If utilising the reset_parameters method, this is not necessary.
    # Additionaly, rng_salt has to be greater than 0 if set since the rng seed cannot be smaller than 0.
    # num_local is the number of local minima that should be computed (corresponding to the number of steps in the search tree that should be performed).
    # max_neighbor is the maximum number of neighbors (of a search tree node) that should be considered without improvement for the found node to be considered a local minimum.
    def __init__(self, num_local: int, max_neighbor: int, num_clusters: int, rng_salt: Optional[int] = None):
        super().__init__()

        # Sanity checks
        assert num_local > 0 and max_neighbor > 0 and num_clusters > 0

        self.num_local = num_local
        self.max_neighbor = max_neighbor
        self.num_clusters = num_clusters

        self.num_samples = -1
        self.num_features = -1

        self.data = None
        self.cluster_labels = None
        self.inertia = -1.0
        self.medoids = np.full(shape = (self.num_clusters,), fill_value = -1, dtype = int)

        # Initialize RNG
        seed = constants.SEED
        if rng_salt is not None:
            assert rng_salt > 0
            seed += rng_salt
        self.rng = np.random.default_rng(seed)

    # Changes the parameters to the passed values and resets all other parameters
    # Explicitely does NOT reset rng in order to allow multiple successive runs with different parameters and RNG. If you want to reset the RNG, you need to create a new clarans instance.
    def set_parameters(self, num_local: Optional[int] = None, max_neighbor: Optional[int] = None, num_clusters: Optional[int] = None) -> None:
        
        # Sanity checks
        if num_local is not None:
            assert num_local > 0
            self.num_local = num_local

        if max_neighbor is not None:
            assert max_neighbor > 0
            self.max_neighbor = max_neighbor

        if num_clusters is not None:
            assert num_clusters > 0
            self.num_clusters = num_clusters

        # Resets parameters
        self.num_samples = -1
        self.num_features = -1

        self.data = None
        self.cluster_labels = None
        self.inertia = float('inf')
        self.medoids = np.full(shape = (self.num_clusters,), fill_value = -1, dtype = int)

    # Runs the CLARANS algorithm on the dataset data using the parameters established in the constructor/set_parameters method
    # data: numpy array of shape (num_samples, feature_dim)
    def fit(self, data: np.array) -> None:

        self.num_samples, self.num_features = data.shape
        self.data = data
        self.cluster_labels = np.full(shape = (self.num_samples,), fill_value = -1, dtype = int)

        self.inertia = float('inf')

        for _ in range(self.num_local):

            # Select num_clusters medoids randomly (corresponds to selecting a random node in the search tree)
            current = self.rng.integers(low = 0, high = self.num_samples, size = self.num_clusters, dtype = int)

            cur_neighbors_visited = 0

            while cur_neighbors_visited < self.max_neighbor:

                # Generate a random neighbor of current
                replace_idx = self.rng.integers(low = 0, high = self.num_clusters, size = 1).item()
                replace_val = None

                # assert that no two medoids can be the same vectors even if a vector appears multiple times in the dataset
                valid = False
                while not valid:
                    replace_val = self.rng.integers(low = 0, high = self.num_samples, size = 1).item()
                    for row in self.data[current,:]:
                        valid = True
                        if np.array_equal(self.data[replace_val,:], row):
                            valid = False

                neighbor = np.array(current)
                neighbor[replace_idx] = replace_val

                if self.calculate_cost_differential(current = current, neighbor = neighbor) < 0:
                    # replace current with the neighbor candidate
                    current = neighbor
                    cur_neighbors_visited = 0
                else:
                    cur_neighbors_visited += 1

            # compare the generated result to the current best result and replace if better (based on inertia)
            labels, inertia = self.generate_cluster_labels(medoids = current)
            if inertia < self.inertia:
                self.cluster_labels = labels
                self.medoids = current
                self.inertia = inertia

    
    # Calculates the cost of a jump, if it is below 0, the neighbor is a better set of medoids than current
    def calculate_cost_differential(self, current: np.array, neighbor: np.array) -> float:

        # m and p are consistent with the notation of the original paper
        # There should be only one entry in the non_zero tuple
        idx_to_swap = np.nonzero(current - neighbor)[0].item()
        m = current[idx_to_swap].item()
        p = neighbor[idx_to_swap].item()

        cost = 0.0

        for j in tqdm.tqdm(range(self.num_samples), total = self.num_samples):

            cur_cluster_id, cur_second_cluster_id = self.get_closest_medoid_idx(medoids = current, sample_idx = j, second_idx = True)

            if cur_cluster_id == idx_to_swap:

                j_two = current[cur_second_cluster_id].item()
                second_cluster_dist = self.get_distance(j, j_two)
                idx_to_swap_new_dist = self.get_distance(j, p)

                if second_cluster_dist <= idx_to_swap_new_dist:
                    # (Case 1): idx is more similar to the 2nd most similar medoid than the proposed new medoid 
                    cost += second_cluster_dist - self.get_distance(j, m)
                else:
                    # (Case 2): idx is more similar to the proposed new medoid than the 2nd most similar medoid
                    cost += idx_to_swap_new_dist - self.get_distance(j, m)
            else:
                
                j_two = current[cur_cluster_id].item()
                cluster_dist = self.get_distance(j, j_two)
                idx_to_swap_new_dist = self.get_distance(j, p)

                if cluster_dist <= idx_to_swap_new_dist:
                    # (Case 3): idx is more similar to its current medoid (which is not proposed to be replaced) than the new proposed medoid
                    cost += 0
                else:
                    # (Case 4): idx is more similar to the new proposed medoid than its current medoid
                    cost += idx_to_swap_new_dist - cluster_dist

        return cost


    def get_distance(self, a_idx: int, b_idx: int) -> float:
        return np.linalg.norm(x = (self.data[a_idx,:] - self.data[b_idx,:]))

    # returns the most similar and the second most similar medoid to the point sample_idx
    def get_closest_medoid_idx(self, medoids: np.array, sample_idx: int, second_idx: bool = False) -> int | Tuple[int, int]:

        dissimilarities = np.linalg.norm(x = (self.data[medoids,:] - self.data[sample_idx,:]), axis = 1)
        idx = np.argmin(dissimilarities)
        if second_idx:
            second_idx = np.argmin(np.delete(dissimilarities, idx))
            return int(idx), int(second_idx)
        else:
            return idx
    
    # Returns an array of shape (num_samples,) with the cluster_ids and the inertia of the calculated labels
    def generate_cluster_labels(self, medoids: np.array) -> Tuple[np.array, float]:
        
        assert self.data is not None
        assert medoids is not None

        result = np.full(shape = (self.num_samples,), fill_value = -1, dtype = int)
        inertia = 0.0

        for idx in range(self.num_samples):
            cluster_id = self.get_closest_medoid_idx(medoids, idx)
            result[idx] = cluster_id
            inertia += self.get_distance(idx, medoids[cluster_id].item())

        return result, inertia
