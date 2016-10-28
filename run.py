import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(sys.path[0],'Provided files','scripts'))
from proj1_helpers import *
from helpers import *
from costs import *
from ridge_regression import *
from least_squares_GD import *
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

a = np.sum(tX==-999,axis=0)
bc = np.nonzero(a>tX.shape[0]/2)
tX = np.delete(tX,bc,axis=1)
print(tX.shape)

y = (y+1)/2

means = np.zeros(tX.shape[1])

for i in range(tX.shape[1]):
    means[i] = np.mean(tX[(tX[:,i] != -999),i],axis=0)
    tX[tX[:, i] == -999, i] = means[i]

poly_tX = build_degree2_poly(tX)

seed = 234
k_fold = 10
lambdas = np.logspace(-6, -1, 50)
# split data in k fold
k_indices = build_k_indices(y, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_te = np.zeros((len(lambdas), k_fold))
rmse_tr = np.zeros((len(lambdas), k_fold))

for k in range(k_fold):
    print("Computing fold no {}".format(k))
    for ind, lamb in enumerate(lambdas):
        y_te = y[k_indices[k]]
        tX_te = poly_tX[k_indices[k]]
        tr_indices = np.delete(k_indices, k, axis=0)
        y_tr = y[tr_indices.flatten()]
        tX_tr = poly_tX[tr_indices.flatten()]

        w = ridge_regression(y_tr, tX_tr, lamb)

        loss_te = compute_loss(y_te, tX_te, w)
        rmse_te[ind][k] = np.sqrt(2 * loss_te)
        loss_tr = compute_loss(y_tr, tX_tr, w)
        rmse_tr[ind][k] = np.sqrt(2 * loss_tr)
        # rmse_te[ind][k] = error_percent(y_te, tX_te, w)
        # rmse_tr[ind][k] = error_percent(y_tr, tX_tr, w)


mean_rmse_te = np.mean(rmse_te, axis=1).flatten()
mean_rmse_tr = np.mean(rmse_tr, axis=1).flatten()

print(mean_rmse_te.shape)

plt.semilogx(lambdas,mean_rmse_te,'-*')
plt.show()

ind = np.argmin(mean_rmse_te)
lambda_ = lambdas[ind]

weights = ridge_regression(y, poly_tX, lambda_)

min_error = error_percent(y,poly_tX,weights)
print("Min rmse = {r} at lambda = {l}".format(r=min_error,l=lambda_))


DATA_TEST_PATH = '../Project1_Data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test = np.delete(tX_test,bc,axis=1)
for i in range(tX_test.shape[1]):
    tX_test[tX_test[:, i] == -999, i] = means[i]

tX_test = build_degree2_poly(tX_test)

OUTPUT_PATH = '../Project1_Data/results.csv'
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)