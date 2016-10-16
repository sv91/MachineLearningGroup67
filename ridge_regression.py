import numpy as np

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""

    N = tx.shape[0]
    M = tx.shape[1]
    A_p = tx.T.dot(tx) + lamb * np.eye(M)
    y_p = tx.T.dot(y);
    w = np.linalg.solve(A_p, y_p)

    return w