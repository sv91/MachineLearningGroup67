# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np
from proj1_helpers import *


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)


def error_percent(y, tx, w):

    y_pred = predict_labels(w,tx)

    return np.mean(y_pred != y)