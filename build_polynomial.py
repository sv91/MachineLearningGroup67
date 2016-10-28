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
    print(max(x_st.flatten()))

    M = x_st.shape[1]

    phi = np.empty([N,int(M*(M+1)/2)])

    a = (M+1)/2

    for i in range(int(np.floor(a))):
        phi[:, M*i:M*(i+1)] = x_st*np.roll(x_st,i,axis=1)

    if M % 2 == 0:
        phi[:,np.floor(a)*M:] = x_st[:,0:int(M/2)]*x_st[:,int(M/2):]

    # print(np.linalg.matrix_rank(phi))

    return phi

def build_degrees_nm(x,n,m):
    N = x.shape[0]
    M = x.shape[1]

    x_st = standardize(x)[0]
    x_st = x_st[:,1:]

    phi = np.zeros((N,M*(m-n+1)))

    for i in range(n,m):
        j = i-n
        phi[:,M*j:M*(j+1)] = x_st**i

    return phi
