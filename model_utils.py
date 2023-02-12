import numpy as np
import scipy as sc
from scipy import sparse as sp

import torch


def inverse_degree_array(d):
    with np.errstate(divide='ignore'):
        invd = 1. / d
    invd[np.isinf(invd)] = 0.
    return invd


def biadjacency_to_laplacian(B, C):
    # 1/2 ( I + D^-1/2 A D^-1/2)
    m, n = B.shape
    A = sp.vstack([sp.hstack([C, B]), sp.hstack([B.T, sp.csc_matrix((n, n))])])
    d = np.array(A.sum(0)).squeeze()
    sqrt_inv_d = inverse_degree_array(d) ** .5
    diag_mat = sp.diags(sqrt_inv_d)
    L = (diag_mat @ A @ diag_mat + sp.eye(m + n)) / 2
    return L


def biadjacency_to_propagation(B, C):
    d_item = np.array(B.sum(0)).squeeze()
    d_user = np.array(B.sum(1)).squeeze()
    d_user_user = np.array(C.sum(0)).squeeze()
    invd_item = inverse_degree_array(d_item)
    invd_user = inverse_degree_array(d_user)
    invd_user_user = inverse_degree_array(d_user_user)
    B_i2u = sp.diags(invd_user) @ B
    B_u2i = sp.diags(invd_item) @ B.T
    B_u2u = sp.diags(invd_user_user) @ C
    return B_i2u, B_u2i, B_u2u


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)