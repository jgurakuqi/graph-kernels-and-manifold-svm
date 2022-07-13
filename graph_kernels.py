from sklearn.model_selection import cross_val_score
from sklearn import manifold, svm
from multiprocessing import Pool


from numpy import (
    mean,
    fill_diagonal,
    reshape,
    float,
    empty,
    object,
    sum,
    maximum,
    max,
    dot,
    transpose,
    concatenate,
    zeros,
)

from numpy.linalg import norm


class cross_validation_analysis(object):
    def __init__(
        self,
        manifold_technique,
        estimator,
        distance_matrices,
        labels,
        cv_method,
        dataset_name,
    ):
        self.manifold_technique = manifold_technique
        self.estimator = estimator
        self.distance_matrices = distance_matrices
        self.labels = labels
        self.cv_method = cv_method
        self.dataset_name = dataset_name

    def __call__(self, act_params):
        return [
            round(
                cross_val_score(
                    estimator=self.estimator,
                    X=self.manifold_technique(
                        n_neighbors=act_params[0], n_components=act_params[1]
                    ).fit_transform(self.distance_matrices),
                    y=self.labels,
                    cv=self.cv_method,
                    n_jobs=-1,
                ).mean(),
                3,
            ),  # Mean -0
            self.dataset_name,  # Dataset Name -1
            act_params[0],  # Neighbours -2
            act_params[1],  # Components -3
            self.manifold_technique.__name__,  # Manifold method name -4
        ]


class shortest_path_kernel:
    def initialize_paths(self, graph):
        dist = graph
        dist[dist == 0] = float("inf")  # Lowest float value
        fill_diagonal(dist, 0)
        return dist

    def compute_floyd_warshall(self, graph):
        graph = graph.astype(float)
        dist = self.initialize_paths(graph)
        v = graph.shape[0]
        for k in range(v):
            for i in range(v):
                for j in range(i, v):
                    dist[i, j] = dist[j, i] = min(dist[i, j], dist[i, k] + dist[k, j])
        return dist

    def compute_shortest_paths(self, graphs):
        return [self.compute_floyd_warshall(adj_matrix) for adj_matrix in graphs]

    def compute_shortest_paths_multi_process(self, graphs, process_pool_size=None):
        results = []
        with Pool(process_pool_size) as executor:
            results = executor.map(func=self.compute_floyd_warshall, iterable=graphs)
        return results

    def compute_similarity(self, first_path, second_path, delta):
        vect1, vect2 = empty([delta + 1, 1]), empty([delta + 1, 1])
        for i in range(delta + 1):
            vect1[i] = sum(first_path == i)
            vect2[i] = sum(second_path == i)
        return dot(
            transpose(vect1 / norm(vect1)),
            vect2 / norm(vect2),
        )[0]

    # similarity between paths weights
    def k_path_weigth(self, first_shortest_path, second_shortest_path):
        v1 = first_shortest_path.shape[0]
        v2 = second_shortest_path.shape[0]
        max_size = maximum(v1, v2) + 1
        75462
        WS1_rows = concatenate(
            [sum(first_shortest_path, axis=1), zeros(max_size - v1)]
        )  # pad with zeros
        WS2_rows = concatenate(
            [sum(second_shortest_path, axis=1), zeros(max_size - v2)]
        )  # pad with zeros
        return dot(WS1_rows, transpose(WS2_rows)) / (norm(WS1_rows) * norm(WS2_rows))

    def compute_similarities_task(self, matrix_id):
        return [
            self.compute_similarity(
                self.all_matrices[matrix_id],
                self.all_matrices[i],
                int(
                    maximum(
                        max(self.all_matrices[matrix_id]), max(self.all_matrices[i])
                    )
                ),
            )
            for i in range(len(self.all_matrices))
        ]

    def compute_similarities_multi_process(self, adj_matrices, process_pool_size=None):
        num_of_matrices = len(adj_matrices)
        all_matrices_ids = range(len(adj_matrices))
        self.all_matrices = adj_matrices
        result = []
        with Pool(process_pool_size) as executor:
            result = executor.map(
                func=self.compute_similarities_task,
                iterable=all_matrices_ids,
            )
        result = reshape(result, (num_of_matrices, num_of_matrices))
        return result
