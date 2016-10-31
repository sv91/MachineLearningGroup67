# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
import sys, os
sys.path.append(os.path.join(sys.path[0],'Provided files','scripts'))
from helpers import standardize

def build_degree2_poly (x):
    """Creates the second degree polynomial basis function for input data x"""

    N = x.shape[0]

    x_st = standardize(x)[0] #input is standardized beforehand

    M = x_st.shape[1]

    phi = np.empty([N,int(M*(M+1)/2)])

    a = (M+1)/2

    for i in range(int(np.floor(a))):
        phi[:, M*i:M*(i+1)] = x_st*np.roll(x_st,i,axis=1)

    if M % 2 == 0:
        phi[:,np.floor(a)*M:] = x_st[:,0:int(M/2)]*x_st[:,int(M/2):]

    return phi

def build_degrees_nm(x,n,m):
    """ Creates the powers of features from nth power to mth"""
	N = x.shape[0]
    M = x.shape[1]

    x_st = standardize(x)[0]
    x_st = x_st[:,1:]

    phi = np.zeros((N,M*(m-n+1)))

    for i in range(n,m):
        j = i-n
        phi[:,M*j:M*(j+1)] = x_st**i

    return phi