import numpy as np
import time

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/ (1 + np.exp(-t))


def calculate_neg_loglike(y, tx, w):
    """compute the cost by negative log likelihood."""

    a = np.log(1 + np.exp(tx.dot(w))).flatten()
    b = (y * (tx.dot(w))).flatten()

    return np.sum(a - b)


def get_oracle(y, tx, w, get_H = True):
    """return the loss, gradient, and hessian."""
    y_e = tx.dot(w).flatten()

    loss = np.sum(np.log(1 + np.exp(y_e)) - (y * y_e))

    grad = tx.T.dot(sigmoid(y_e) - y)

    if get_H:
        S = np.diag((sigmoid(y_e) * (1 - sigmoid(y_e))))

        H = tx.T.dot(S.dot(tx))
    else:
        H = np.ones((tx.shape[1],tx.shape[1]))


    return loss, grad, H

def get_oracle_penalized(y, tx, w, lambda_, get_H = True):
    """return the loss, gradient, and hessian."""
    y_e = tx.dot(w).flatten()

    loss = np.sum(np.log(1 + np.exp(y_e)) - (y * y_e)) + lambda_*np.linalg.norm(w)**2

    grad = tx.T.dot(sigmoid(y_e) - y) + 2*lambda_*w

    if get_H:
        S = np.diag((sigmoid(y_e) * (1 - sigmoid(y_e))))

        H = tx.T.dot(S.dot(tx)) + 2*lambda_*np.eye(S.shape)
    else:
        H = np.ones((tx.shape[1],tx.shape[1]))


    return loss, grad, H

def logistic_regression_newton(y, tx, gamma, max_iter,tol = 1e-8):
    # init parameters
    losses = []

    # build tx
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, grad_L, H = get_oracle(y, tx, w)

        w = w - gamma * np.linalg.inv(H).dot(grad_L)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < tol:
            break

    return w


def logistic_regression(y,tx, gamma, max_iter, tol = 1e-8):
    # init parameters
    losses = []

    # build tx
    w = np.zeros(tx.shape[1])

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, grad_L, H = get_oracle(y, tx, w, get_H=False)

        w = w - gamma * grad_L
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < tol:
            break

    return w

def logistic_regression_penalized(y,tx, gamma, lambda_, max_iter, tol = 1e-8):
    # init parameters
    losses = []

    # build tx
    w = np.zeros(tx.shape[1])
    time.clock()
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, grad_L, H = get_oracle_penalized(y, tx, w, lambda_, get_H=False)

        w = w - gamma * grad_L
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
            print(time.clock(), "seconds elapsed")

        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < tol:
            break

    return w

def logistic_regression_penalized_newton(y, tx, gamma, max_iter, lambda_, tol = 1e-8):
    # init parameters
    losses = []

    # build tx
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, grad_L, H = get_oracle_penalized(y, tx, w, lambda_)

        w = w - gamma * np.linalg.inv(H).dot(grad_L)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < tol:
            break

    return w