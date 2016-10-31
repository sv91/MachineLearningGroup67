import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.append(os.path.join(sys.path[0],'Provided files','scripts'))
from helpers import *
from costs import *
from implemetations import *
from build_polynomial import *


def build_k_indices (y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

DATA_TRAIN_PATH = '../Project1_Data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# a = np.sum(tX==-999,axis=0)
# bc = np.nonzero(a>tX.shape[0]/2)
# tX = np.delete(tX,bc,axis=1)
# print(tX.shape)

means = np.zeros(tX.shape[1])

for i in range(tX.shape[1]):
    means[i] = np.mean(tX[(tX[:,i] != -999),i],axis=0)
    tX[tX[:, i] == -999, i] = means[i]

poly_tX = build_degree2_poly(tX)
tx_3_10 = build_degrees_nm(tX,3,10)
poly_tX = np.hstack((poly_tX,tx_3_10))

print(max(poly_tX.flatten()))
std_tX = np.std(poly_tX, axis=0)
poly_tX[:, std_tX>0] = poly_tX[:, std_tX>0] / std_tX[std_tX>0]
print(max(poly_tX.flatten()))
y = (y+1)/2

print(poly_tX.shape)

seed = 234
k_fold = 4
# split data in k fold
k_indices = build_k_indices(y, k_fold, seed)
# define lists to store the loss of training data and test data
# rmse_te = np.zeros(k_fold)
# rmse_tr = np.zeros(k_fold)
error_te = np.zeros(k_fold)
error_tr = np.zeros(k_fold)
max_iter = 3000
lambda_ = 100

for k in range(k_fold):
    print("Computing fold no {}".format(k+1))
    y_te = y[k_indices[k]]
    tX_te = poly_tX[k_indices[k]]
    tr_indices = np.delete(k_indices, k, axis=0)
    y_tr = y[tr_indices.flatten()]
    tX_tr = poly_tX[tr_indices.flatten()]

    L = 1/4*np.linalg.norm(tX_tr.T.dot(tX_tr),ord=2)+2*lambda_
    print(L)
    gamma = 1/L
    print(gamma)
    # gamma = 1e-25

    w = logistic_regression_penalized(y_tr, tX_tr, gamma, lambda_, max_iter)

    # loss_te = compute_loss(y_te, tX_te, w)
    # rmse_te[k] = np.sqrt(2 * loss_te)
    # loss_tr = compute_loss(y_tr, tX_tr, w)
    # rmse_tr[k] = np.sqrt(2 * loss_tr)
    error_te[k] = error_percent(2*y_te-1, tX_te, w)
    error_tr[k] = error_percent(2*y_tr-1, tX_tr, w)

    print("Fold no {} : te error = {} , tr error = {} ".format(k+1,error_te[k],error_tr[k]))


# plt.plot(range(k_fold)+1,error_te,'b')
# plt.plot(range(k_fold)+1,error_te,'r')

# ind = np.argmin(error_te)

# tr_indices = np.delete(k_indices, k, axis=0)
# y_tr = y[tr_indices.flatten()]
# tX_tr = poly_tX[tr_indices.flatten()]

L = 1/4*np.linalg.norm(poly_tX.T.dot(poly_tX),ord=2)
gamma = 1/L

weights = reg_logistic_regression(y, poly_tX, lambda_, max_iter, gamma)

DATA_TEST_PATH = '../Project1_Data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print(tX_test.shape)

for i in range(tX_test.shape[1]):
    tX_test[tX_test[:, i] == -999, i] = means[i]

N = tX_test.shape[0]
y_pred = np.zeros(N)

#divide test into pieces and continue with classifying
n_of_piece = 10
div_size = int(np.ceil(N/n_of_piece))

for i in range(n_of_piece):
    print('Doing part {} of classification'.format(i+1))

    fin = min(div_size*(i+1),N)
    tX_test_i = tX_test[div_size*i:fin,:]
    # tX_test = np.delete(tX_test,bc,axis=1)

    poly_tX_test_i = build_degree2_poly(tX_test_i)
    tx_3_10 = build_degrees_nm(tX_test_i,3,10)
    tX_test_i = np.hstack((poly_tX_test_i,tx_3_10))
    std_tX = np.std(tX_test_i, axis=0)
    tX_test_i[:, std_tX>0] = tX_test_i[:, std_tX>0] / std_tX[std_tX>0]

    y_pred_i = predict_labels(weights, tX_test_i)
    y_pred[div_size*i:fin] = y_pred_i

print(y_pred.shape)

OUTPUT_PATH = '../Project1_Data/results.csv'

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)