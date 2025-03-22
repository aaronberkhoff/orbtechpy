import numpy as np

def unscented_transform(mean: np.array, covariance: np.array, beta: float = 2, alpha=1):

    state_length = mean.shape[0]
    kappa = 3 - state_length

    lower_triangle = np.linalg.cholesky(covariance)
    lam = alpha**2 * (state_length + kappa) - state_length

    sigma_points = np.zeros((state_length, 2 * state_length + 1))
    sigma_points[:, 0] = mean.ravel()

    weight = np.sqrt(state_length + lam) * lower_triangle
    # temp = mean + weight
    sigma_points[:, 1 : state_length + 1] = mean + weight

    sigma_points[:, state_length + 1 :] = mean - weight

    return sigma_points

def unscented_weights(shape, beta: float = 2, alpha: float = 1):

    state_length = shape[0]
    kappa = 3 - state_length
    lam = alpha**2 * (state_length + kappa) - state_length

    weights_mean = np.zeros(shape=shape[1])
    weights_cov = np.zeros(shape=shape[1])

    weights_mean[0] = lam / (state_length + lam)
    weights_cov[0] = weights_mean[0] + (1 - alpha**2 + beta)

    temp = 1 / (2 * (state_length + lam))

    weights_mean[1:] = temp
    weights_cov[1:] = temp

    return weights_mean, weights_cov


def reconstruct_sigma_points(sigma_points, weights: tuple = None):

    # mean = np.dot(weights[0],sigma_points)
    mean = np.einsum("i, ki", weights[0], sigma_points)[:, np.newaxis]

    difference = sigma_points - mean

    # covariance = np.einsum('i, ki', weights[1], temp)
    covariance = np.einsum("ij,j,kj -> ik", difference, weights[1], difference)
    # covariance = (np.einsum('i, ki', weights[1], difference)) @ difference.T
    # covariance = difference @ (weights[1][np.newaxis, :] * difference).T

    return mean, covariance