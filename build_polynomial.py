# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
import sys, os
sys.path.append(os.path.join(sys.path[0],'Provided files','scripts'))
from helpers import standardize

def build_degree2_poly (x):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""

    N = x.shape[0]

    x_st = standardize(x)[0]
    # x_st = np.hstack((np.ones((x.shape[0],1)), x))

    M = x_st.shape[1]

    phi = np.empty([N,M*(M+1)/2])

    a = (M+1)/2

    for i in range(int(np.floor(a))):
        phi[:, M*i:M*(i+1)] = x_st*np.roll(x_st,i,axis=1)

    if M % 2 == 0:
        phi[:,np.floor(a)*M:-1] = x_st[:,0:M/2-1]*x_st[:,M/2:-1]

    # print(np.linalg.matrix_rank(phi))

    return phi