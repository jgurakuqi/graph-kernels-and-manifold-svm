from sklearn.model_selection import cross_val_score
from multiprocessing import Pool
from platform import processor


if "x86" in processor():
    from sklearnex import patch_sklearn

    patch_sklearn()
    from sklearn.model_selection import cross_val_score


from numpy import (
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
    ):
        self.manifold_technique = manifold_technique
        self.estimator = estimator
        self.distance_matrices = distance_matrices
        self.labels = labels
        self.cv_method = cv_method

    def __call__(self, act_params):
        return (
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
            ),
            act_params[0],
            act_params[1],
        )


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

    def compute_shortest_paths_multi_process(self, graphs):
        results = []
        with Pool() as executor:
            results = executor.map(func=self.compute_floyd_warshall, iterable=graphs)
        return results

    def compute_delta_similarity(self, first_path, second_path, delta):
        vect1, vect2 = empty([delta + 1, 1]), empty([delta + 1, 1])
        for i in range(delta + 1):
            vect1[i] = sum(first_path == i)
            vect2[i] = sum(second_path == i)
        return dot(
            transpose(vect1 / norm(vect1)),
            vect2 / norm(vect2),
        )[0]

    def compute_similarities_task(self, matrix_id):
        return [
            self.compute_delta_similarity(
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

    def compute_similarities_multi_process(self, adj_matrices):
        num_of_matrices = len(adj_matrices)
        all_matrices_ids = range(len(adj_matrices))
        self.all_matrices = adj_matrices
        result = []
        with Pool() as executor:
            result = executor.map(
                func=self.compute_similarities_task,
                iterable=all_matrices_ids,
            )
        return reshape(result, (num_of_matrices, num_of_matrices))
