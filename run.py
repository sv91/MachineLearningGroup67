import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(sys.path[0],'Provided files','scripts'))
from proj1_helpers import *
from helpers import *
from costs import compute_loss
from ridge_regression import *
from least_squares_GD import *

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

tX = standardize(tX)[0]

seed = 100
k_fold = 10
lambdas = np.linspace(0.001, 0.01, 100)
# split data in k fold
k_indices = build_k_indices(y, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_te = np.zeros((len(lambdas), k_fold))


for k in range(k_fold):
    print("Computing fold no {}".format(k))
    for ind, lamb in enumerate(lambdas):
        y_te = y[k_indices[k]]
        tX_te = tX[k_indices[k]]
        tr_indices = np.delete(k_indices, k, axis=0)
        y_tr = y[tr_indices.flatten()]
        tX_tr = tX[tr_indices.flatten()]

        w = ridge_regression(y_tr, tX_tr, lamb)

        loss_te = compute_loss(y_te, tX_te, w)
        rmse_te[ind][k] = np.sqrt(2 * loss_te)

rmse_te = np.mean(rmse_te, axis=1).flatten()

plt.plot(lambdas,rmse_te,'-*')
plt.show()

ind = np.argmin(rmse_te)
lambda_ = lambdas[ind]

weights = ridge_regression(y, tX, lambda_)

min_error = np.sqrt(2*compute_loss(y,tX,weights))
print("Min rmse = {r} at lambda = {l}".format(r=min_error,l=lambda_))


DATA_TEST_PATH = '../Project1_Data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test = standardize(tX_test)[0]

OUTPUT_PATH = '../Project1_Data/results.csv'
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)