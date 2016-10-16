import numpy as np


def compute_gradient_LS(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    N = len(y)

    grad_f = -1/N*tx.T.dot(e)

    return grad_f


def least_squares_GD(y, tx, gamma, max_iters):
    """Gradient descent algorithm."""

    w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        grad_f = compute_gradient_LS(y, tx, w)

        w = w - gamma * grad_f


    return w