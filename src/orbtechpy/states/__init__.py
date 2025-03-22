import numpy as np
from orbtechpy.utils import unscented_transform, unscented_weights

class SigmaPoints:

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):

        sigma_points = unscented_transform(mean, covariance)
        self.sigma_points = sigma_points
        self._shape = sigma_points.shape
        self._weights = unscented_weights(shape=sigma_points.shape)

    def __iter__(self):
        return iter(self.states)

    def __sub__(self, matrix):

        if not isinstance(matrix, np.ndarray):
            return NotImplemented

        return self.sigma_points - matrix

    def __matmul___(self, matrix):

        if not isinstance(matrix, np.ndarray):
            return NotImplemented

        return self.sigma_points @ matrix

    @property
    def state(self):
        return self.sigma_points[0:6]

    @property
    def noise(self):
        return self.sigma_points[9:]

    @property
    def process_noise(self):
        return self.sigma_points[6:9]

    @property
    def matrix(self):
        return self.sigma_points

    @property
    def weights(self):
        return self._weights

    @property
    def shape(self):
        return self._shape
