#Group 67

import numpy as np
import matplotlib.pyplot as plt
import time
import costs #sprawdzic biblioteke

from numpy.random import rand, randn


#-----------------------------------------------------------------------
#                           Least Squares GD


def compute_gradient_LS(y, tx, w):
    """Compute the gradient."""
     e = y - tx.dot(w)
    N = len(y)

    grad_f = -1/N*tx.T.dot(e)

    return grad_f


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w=initial_w
    for n_iter in range(max_iters):
        
        grad_f = compute_gradient_LS(y, tx, w)
        w = w - gamma * grad_f
        loss = 


    return (w,loss)                              #DOPISAĆ BŁĄD WSZĘDZIE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


#-----------------------------------------------------------------------
#                           Least Squares SGD

def computeGradient(y,tx,w)
    """Compute the gradient."""
    e = y - tx.dot(w)
    return - tx.T * e

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w=initial_w
    for i in range(max_iters):
        r = np.random.random_integers(0,y.shape[0])
        ln = computeGradient(y[r],tx[r,:],w)
        w = w - gamma * ln
        loss = 
    return (w,loss)



#-----------------------------------------------------------------------
#                           Least Squares


def least_squares(y, tx):
    """Compute least squares regression"""
    xtx=np.dot(tx.T,tx)
    xtx=np.linalg.inv(xtx)
    w=np.dot(xtx,np.dot(tx.T,y))
    
    loss=
    
    return (w,loss)



#-----------------------------------------------------------------------
#                           Ridge Regression

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    N = tx.shape[0]
    M = tx.shape[1]
    A_p = tx.T.dot(tx) + lambda_ * np.eye(M)
    y_p = tx.T.dot(y);
    w = np.linalg.solve(A_p, y_p)
    
    loss=
    return (w,loss)

#-----------------------------------------------------------------------
#                           Logistic Regression


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

def logistic_regression(y,tx, initial_w,  max_iter, gamma):
    """implement logistic regression"""
    
    # init parameters
    tol = 1e-8
    losses = []

    # build tx
    w=initial_w

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

    return (w,losses[-1])




#-----------------------------------------------------------------------
#                   Regularized Logistic Regression

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

def reg_logistic_regression(y,tx, lambda_, max_iter, gamma):
    """implement regularized logistic regression"""
    # init parameters
    losses = []
    tol=1e-8
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

    return (w,losses[-1])

#-------------------------------------------------------------------------------------------------------------------------------
