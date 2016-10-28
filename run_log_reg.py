import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.append(os.path.join(sys.path[0],'Provided files','scripts'))
from proj1_helpers import *
from helpers import *
from costs import *
from ridge_regression import *
from least_squares_GD import *
from build_polynomial import *
from logistic_regression import *

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
print(max(tX.flatten()))
std_tX = np.std(tX, axis=0)
tX[:, std_tX>0] = tX[:, std_tX>0] / std_tX[std_tX>0]
# tx_3_10 = build_degrees_nm(tX,3,10)
# poly_tX = np.hstack((poly_tX,tx_3_10))
print(max(tX.flatten()))
y = (y+1)/2


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
lambda_ = 10

for k in range(k_fold):
    print("Computing fold no {}".format(k))
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

    print("Fold no {} : tr error = {} , te error = {} ".format(k,error_te[k],error_tr[k]))


# plt.plot(range(k_fold)+1,error_te,'b')
# plt.plot(range(k_fold)+1,error_te,'r')

# ind = np.argmin(error_te)

# tr_indices = np.delete(k_indices, k, axis=0)
# y_tr = y[tr_indices.flatten()]
# tX_tr = poly_tX[tr_indices.flatten()]

L = 1/4*np.linalg.norm(poly_tX.T.dot(poly_tX),ord=2)
gamma = 1/L

weights = logistic_regression_penalized(y, poly_tX, gamma, lambda_, max_iter)

DATA_TEST_PATH = '../Project1_Data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# tX_test = np.delete(tX_test,bc,axis=1)
for i in range(tX_test.shape[1]):
    tX_test[tX_test[:, i] == -999, i] = means[i]

tX_test = build_degree2_poly(tX_test)

OUTPUT_PATH = '../Project1_Data/results.csv'
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)