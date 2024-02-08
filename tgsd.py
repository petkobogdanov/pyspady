import cmath
# import itertools
import math
from math import gcd, pi
import numpy as np
import scipy.io
import tensorly as tl
import sparse
from tensorly.contrib.sparse import tensor as sp_tensor
import scipy.sparse as sp
from scipy.fftpack import fft
from scipy.stats import linregress
import time
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D

from Y_unittest import TestYConversion
from W_unittest import TestWConversion

from collections import defaultdict
from scipy.sparse import spmatrix

import json
import pandas as pd
import random
import csv


# from numba import njit, prange, jit
# from numba.typed import List
# max number of pairs defined
num_pairs = 100

# maximum row and column indices
max_index = 20

# Generate random coordinate pairs
coordinates = [(random.randint(1, max_index), random.randint(1, max_index)) for _ in range(num_pairs)]

# Write the coordinates to a CSV file
with open('graph_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['R', 'C'])  # Write header row
    writer.writerows(coordinates)  # Write coordinate pairs


# Read the CSV file into a pandas DataFrame
df = pd.read_csv('graph_data.csv')
#df_values = df.iloc[1:]

# Shape is determined by max (should be set in config because the max shape may not necessarily be present)
max_row = df['R'].max()
max_col = df['C'].max()

# empty dense matrix filled with zeros (can be converted to sparse matrix for operations)
dense_matrix = np.zeros((max_row, max_col))

# fill dense matrix
for _, row in df.iterrows():
    dense_matrix[row['R'] - 1, row['C'] - 1] = 1  # Subtract 1 to convert to zero-based indexing

print(dense_matrix)

def load_matrix() -> dict:
    """
    Loads matrix from file in directory
    Returns:
        dict: dictionary with variable names as keys, and loaded matrices as values
    """
    return scipy.io.loadmat('demo_data.mat')


def load_syn_data() -> dict:
    """
    Loads synthetic matrix from file in directory
    Returns:
        dict: dictionary with variable names as keys, and loaded matrices as values
    """
    return scipy.io.loadmat('syn_data.mat')


def gen_gft(p_dict: dict, is_normalized: bool) -> list[np.ndarray]:
    """
    Constructs a PsiGFT from matlab dictionary (for now)
    Args:
        p_dict (dict): Given matlab dictionary
        is_normalized (bool): Whether the matrix should be normalized
    Returns:
        list[np.ndarray]: list of numpy arrays in form [psi_gft, eigenvalues]
    """
    adj = p_dict['adj']  # given adj matrix
    # print(np.linalg.matrix_rank(adj.toarray()))
    # calculate sum along columns
    D = np.diag(np.array(adj.sum(axis=0)).flatten())
    # print(np.linalg.matrix_rank(D.toarray()))
    L = D - adj  # Laplacian matrix
    # normalize eigenvectors = D^-1/2*L*D^-1/2
    if is_normalized:
        D_sqrt_inv = sp.diags(1.0 / np.sqrt(np.array(D.sum(axis=0)).flatten()))
        new_L = D_sqrt_inv @ L @ D_sqrt_inv
        eigenvalues, psi_gft = np.linalg.eigh(new_L.toarray())
        idx = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        psi_gft = psi_gft[:, idx]
        return [psi_gft, eigenvalues]
    # print(np.linalg.matrix_rank(L.toarray()))
    eigenvalues, psi_gft = np.linalg.eigh(L, UPLO='U')
    # sort eigenvalues by ascending order such that constant vector is in first cell
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    psi_gft = psi_gft[:, idx]
    psi_gft = np.squeeze(np.asarray(psi_gft))
    return [psi_gft, eigenvalues]

def gen_gft_new(matrix: spmatrix, is_normalized: bool) -> list[np.ndarray]:
    """
    Constructs a PsiGFT from matlab dictionary (for now)
    Args:
        p_dict (dict): Given matlab dictionary
        is_normalized (bool): Whether the matrix should be normalized
    Returns:
        list[np.ndarray]: list of numpy arrays in form [psi_gft, eigenvalues]
    """
    adj = matrix  # given adj matrix
    # user can supply graph in matrix form or binary form (R, C, V) or (R, C)
    # print(np.linalg.matrix_rank(adj.toarray()))
    # calculate sum along columns
    D = np.diag(np.array(adj.sum(axis=0)).flatten())
    # print(np.linalg.matrix_rank(D.toarray()))
    L = D - adj  # Laplacian matrix
    # normalize eigenvectors = D^-1/2*L*D^-1/2
    if is_normalized:
        D_sqrt_inv = sp.diags(1.0 / np.sqrt(np.array(D.sum(axis=0)).flatten()))
        new_L = D_sqrt_inv @ L @ D_sqrt_inv
        eigenvalues, psi_gft = np.linalg.eigh(new_L.toarray())
        idx = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        psi_gft = psi_gft[:, idx]
        return [psi_gft, eigenvalues]
    # print(np.linalg.matrix_rank(L.toarray()))
    eigenvalues, psi_gft = np.linalg.eigh(L, UPLO='U')
    # sort eigenvalues by ascending order such that constant vector is in first cell
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    psi_gft = psi_gft[:, idx]
    psi_gft = np.squeeze(np.asarray(psi_gft))
    return [psi_gft, eigenvalues]

def gen_dft(t: int) -> np.ndarray:
    """
    Constructs a PsiDFT
    Args:
        t (int): Number of timesteps
    Returns:
        np.ndarray: new DFT matrix
    """
    return (1 / np.sqrt(t)) * fft(np.eye(t))


def gen_rama(t: int, max_period: int):
    """
    Constructs a Ramanujan periodic dictionary
    Args:
        t (int): Number of timesteps
        max_period (int): Number of max periods

    Returns:
        Ramanujan periodic dictionary of t timesteps (rows) with max_period
    """
    A = np.array([])

    for n in range(1, max_period + 1):  # 1:max_period
        c1 = np.zeros((n, 1), dtype=complex)
        k_orig = np.arange(1, n + 1)
        k = [k__ for k__ in k_orig if gcd(k__, n) == 1]
        for n_ in range(n):  # goes up to n-1 inclusive
            for a in k:
                c1[n_] = c1[n_] + cmath.exp(1j * 2 * pi * a * n_ / n)
        c1 = np.real(c1)

        k_orig = np.arange(1, n + 1)
        k = [k__ for k__ in k_orig if gcd(k__, n) == 1]
        CN_col_size = len(k)  # size(k, 2) -> number of columns

        shifted_columns = []

        for j in range(1, CN_col_size + 1):
            shift = np.roll(c1, j - 1)
            shifted_columns.append(shift)
        # concatenate along the vertical axis
        CN = np.concatenate(shifted_columns, axis=1)

        # CNA = repmat(CN,floor(rowSize/N),1)
        num_repeat = t // n
        CNA = np.tile(CN, (num_repeat, 1))
        # CN_cutoff = CN(1:rem(rowSize,N),:);
        remainder = t % n
        CN_cutoff = CN[:remainder, :]
        # CNA = cat(1,CNA,CN_cutoff);
        CNA = np.concatenate((CNA, CN_cutoff), axis=0)
        # A=cat(2,A,CNA);
        if A.size == 0:
            A = CNA
        else:
            A = np.concatenate((A, CNA), axis=1)
    return A


def tgsd(X, psi_d, phi_d, mask,
         iterations: int, k: int, lambda_1: int, lambda_2: int, lambda_3: int, rho_1: int, rho_2: int, type: str):
    """
    Decomposes a temporal graph signal as a product of two fixed dictionaries and two corresponding sparse encoding matrices
    Args:
        X: Temporal graph signal input
        psi_d: Some graph dictionary, Ψ
        phi_d: Some time series dictionary,
        mask:
        iterations:
        k:
        lambda_1:
        lambda_2:
        lambda_3:
        rho_1:
        rho_2:
        type:

    Returns:

    """

    def is_orthonormal(p_psi_or_phi):
        """
        Determines if graph or time series dictionary is orthonormal
        Args:
            p_psi_or_phi: Specified graph or time series dictionary
        Returns:
            True if orthonormal, false otherwise
        """
        return np.allclose(np.dot(p_psi_or_phi.T.conj(), p_psi_or_phi), np.eye(p_psi_or_phi.shape[1])) and \
               np.allclose(np.linalg.norm(p_psi_or_phi, axis=0), np.ones(p_psi_or_phi.shape[1]))

    def update_d(p_P, p_X, p_mask, p_lambda_3):
        """
        Learns D via P, X, mask, and lambda 3
        Args:
            p_P: ΨYWΦ
            p_X: Some temporal graph signal
            p_mask: Indices represent masked indices in D
            p_lambda_3: Some sparsity regularization parameter, λ3
        Returns:
            D = (P +𝜆3Ω ⊙ X) ⊘ (I +𝜆3Ω)
        """
        p_mask, missing_mask, observed_mask = p_mask - 1, np.zeros(p_P.shape), np.ones(p_P.shape)
        missing_mask[p_mask % p_P.shape[0], p_mask // p_P.shape[0]] = 1
        return (p_P.astype(np.complex128) + (p_lambda_3 * (observed_mask - missing_mask)) * p_X.astype(
            np.longdouble)) / (
                       1 + (p_lambda_3 * (observed_mask - missing_mask)))

    def hard_update_y(p_sigma, p_W, p_X, p_Z, p_psi, p_phi, p_lambda_1, p_Q_1, p_gamma_1, p_rho_1):
        """
        Learn Y based on E1: Y = Q1E1QT2.
        Args:
            p_sigma: Some identity matrix, σ
            p_W: Some encoding matrix, W
            p_X: Some temporal graph signal, X
            p_Z: Some intermediate variable, Z
            p_psi: Some dictionary of non-orthogonal atoms, Ψ
            p_phi: Some time dictionary of atoms, Φ
            p_lambda_1: Some diagonal non-negative eigenvalue matrix, λ1
            p_Q_1: Some orthonormal eigenvector matrix, Q1
            p_gamma_1: Some Lagrangian multiplier, Γ1
            p_rho_1: Some penalty parameter, ρ1

        Returns:
            Y = Q1E1QT2
        """
        _B = p_sigma.astype(np.longdouble) @ p_W.astype(np.complex128) @ p_phi.astype(np.complex128)
        _lambda_2, Q_2 = np.linalg.eigh((_B @ _B.conj().T), UPLO='U')
        _lambda_2 = np.diag(_lambda_2)
        # Pi=2*Psi'*X*B'+rho_1*Z+Gamma_1;
        Pi = 2 * p_psi.astype(np.longdouble).T @ p_X.astype(np.complex128) @ (
            _B.astype(np.complex128)).conj().T + p_rho_1 * p_Z.astype(np.complex128) + p_gamma_1

        QPiQ = p_Q_1.astype(np.longdouble).T @ Pi.astype(np.complex128) @ Q_2.astype(np.complex128)
        diagonal_lambda_1, diagonal_lambda_2 = np.diag(p_lambda_1)[:, None], (np.diag(_lambda_2)[:, None]).T
        temp0 = 2 * diagonal_lambda_1.astype(np.longdouble) @ diagonal_lambda_2.astype(np.longdouble) + rho_2
        E = QPiQ / temp0
        return p_Q_1.astype(np.longdouble) @ E.astype(np.complex128) @ Q_2.astype(np.complex128).conj().T

    def hard_update_w(p_sigma, p_V, p_X, p_Y, p_psi, p_phi, p_lambda_4, p_Q_4, p_gamma_2, p_rho_2):
        """
        Learns W where W = Q3E2QT4, where E2(i, j) = [QT3 Π 2Q4]i,j/2[Λ4]ii[Λ3]jj+p2 and
        (Q3,Λ3 and (Q4,Λ4) are the (eigenvector, eigenvalue) matrices of
        ATA and ΦΦT , respectively.
        Args:
            p_sigma: Some identity matrix, σ
            p_V: Some intermediate variable, V
            p_X: Some temporal graph signal, X
            p_Y: Some encoding matrix, Y
            p_psi: Some dictionary of atoms, Ψ
            p_phi: Some non-orthogonal time dictionary of atoms, Φ
            p_lambda_4: Some diagonal non-negative eigenvalue matrix, λ4
            p_Q_4: Some orthonormal eigenvector matrix, Q4
            p_gamma_2: Some Lagrangian multiplier, Γ2
            p_rho_2: Some penalty parameter, ρ2

        Returns:
            W = Q3E2QT4
        """
        _A = p_psi.astype(np.longdouble) @ p_Y.astype(np.complex128) @ p_sigma.astype(np.longdouble)
        _lambda_3, Q_3 = np.linalg.eigh((_A.conj().T @ _A), UPLO='U')
        _lambda_3 = np.diag(_lambda_3)
        # Pi=2*A'*X*Phi'+rho_2*V+Gamma_2;
        Pi = 2 * (_A.astype(np.complex128)).conj().T @ p_X.astype(np.complex128) @ (
            p_phi.astype(np.complex128)).conj().T + p_rho_2 * p_V + p_gamma_2
        QPiQ = Q_3.astype(np.complex128).conj().T @ Pi @ p_Q_4.astype(np.complex128)

        diagonal_lambda_3, diagonal_lambda_4 = np.diag(_lambda_3)[:, None], (np.diag(p_lambda_4)[:, None]).T
        temp0 = 2 * diagonal_lambda_3.astype(np.longdouble) @ diagonal_lambda_4.astype(np.longdouble) + rho_2
        E = QPiQ / temp0
        return Q_3.astype(np.complex128) @ E.astype(np.complex128) @ Q_4.astype(np.complex128).conj().T

    def get_object(p_mask, p_D, p_X, p_phi, p_psi, p_Y, p_sigma, p_W, p_lambda_1, p_lambda_2, p_lambda_3):
        """
        Returns a new object represented by equation X−YΨWΦ‖
        Args:
            p_mask: Some specified indexed mask
            p_D: Some intermediate variable, D
            p_X: Some temporal graph signal, X
            p_phi: Some time dictionary of atoms Phi
            p_psi: Some dictionary of atoms, Psi
            p_Y: Some encoding matrix Y
            p_sigma: Some identity matrix, σ
            p_W: Some encoding matrix W
            p_lambda_1: Some sparsity regularization parameter, λ1
            p_lambda_2: Some sparsity regularization parameter, λ2
            p_lambda_3: Some sparsity regularization parameter, λ3

        Returns:
            New object represented by X−YΨWΦ‖
        """
        # temp=X(:)-D(:);
        # term3=norm(temp(setdiff(1:end,mask)));
        p_D, p_phi, p_Y, p_W = p_D.astype(np.complex128), p_phi.astype(np.complex128), p_Y.astype(
            np.complex128), p_W.astype(np.complex128)
        p_sigma, p_X, p_psi = p_sigma.astype(np.longdouble), p_X.astype(np.longdouble), p_psi.astype(np.longdouble)

        p_mask, missing_mask, observed_mask = p_mask - 1, np.zeros(p_X.shape), np.ones(p_X.shape)
        missing_mask[p_mask % p_X.shape[0], p_mask // p_X.shape[0]] = 1
        term_3 = np.linalg.norm((observed_mask - missing_mask) * (p_X - p_D), ord=2)
        # obj_new = norm(D-Psi*Y*Sigma*W*Phi)+lambda_1*norm(Y,1)+lambda_2*norm(W,1)+lambda_3*term3;
        return np.linalg.norm(p_D - p_psi @ p_Y @ p_sigma @ p_W @ p_phi, ord=2) \
               + p_lambda_1 * np.linalg.norm(np.abs(p_Y), ord=1) + p_lambda_2 * np.linalg.norm(np.abs(p_W),
                                                                                               ord=1) + p_lambda_3 * term_3

    _, t = X.shape
    hold, Y1 = psi_d.shape
    p, t = phi_d.shape
    W = np.zeros((k, p))
    W[0, 0] = .000001
    if (not is_orthonormal(phi_d) and not is_orthonormal(psi_d)) or (
            is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
        Y = np.eye(Y1, k)
    else:
        Y = np.zeros((Y1, k))
    sigma = np.eye(k, k)
    sigmaR = np.eye(sigma.shape[0], sigma.shape[1])
    V, Z = 0, Y
    gamma_1, gamma_2 = Y, W
    obj_old, objs = 0, []
    I_Y = np.eye((W @ phi_d @ (W @ phi_d).T).shape[0])
    I_W = np.eye(((psi_d @ Y).T @ psi_d @ Y).shape[0])
    # [Q_1,Lam_1]=eig(Psi'*Psi);
    lam_1, Q_1 = np.linalg.eigh((psi_d.T @ psi_d), UPLO='U')
    lam_4, Q_4 = np.linalg.eigh((phi_d.astype(np.complex128) @ phi_d.astype(np.complex128).conj().T), UPLO='U')
    lam_1, lam_4 = np.diag(lam_1), np.diag(lam_4)
    XPhiT = X.astype(np.complex128) @ phi_d.astype(np.complex128).conj().T
    PsiTX = psi_d.astype(np.longdouble).T @ X.astype(np.complex128)

    if mask.any():
        if type == "row":  # row-mask
            n, m = X.shape
            temp2 = np.ones((n, m))
            temp2[mask - 1 % X[0], :] = 0
            mask = np.argwhere(temp2 == 0)
        elif type == "col" or type == "pred":  # column or pred mask
            n, m = X.shape
            temp2 = np.ones((n, m))
            temp2[:, mask - 1] = 0
            mask = np.argwhere(temp2 == 0)

    # plt.figure()

    for i in range(1, 1 + iterations):
        P = (psi_d.astype(np.longdouble) @ Y.astype(np.complex128) @ W.astype(np.complex128) @ phi_d.astype(
            np.complex128))
        D = update_d(P, X, mask, lambda_3)
        #     B=Sigma*W*Phi;
        #     Y=(2*Psi'*D*B'+rho_1*Z+Gamma_1)*inv(2*(B*B')+rho_1*I_y+exp(-15));
        B = sigma.astype(np.longdouble) @ W.astype(np.complex128) @ phi_d.astype(np.complex128)
        if mask.any():
            # Both are orth OR Psi is orth, Phi is not orth
            if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                    is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                Y = (2 * psi_d.conj().T.astype(np.longdouble) @ D.astype(np.complex128) @ B.conj().T + rho_1 * Z.astype(
                    np.complex128) + gamma_1.astype(np.complex128)) @ np.linalg.pinv(
                    2 * (B @ B.conj().T) + rho_1 * I_Y + math.exp(-15)).astype(np.complex128)
            # Phi is orth, Psi is not orth OR Psi is not orth, Phi is not orth
            elif (is_orthonormal(phi_d) and not is_orthonormal(psi_d)) or (
                    not is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                Y = hard_update_y(sigma, W, D, Z, psi_d, phi_d, lam_1, Q_1, gamma_1, rho_1)
        else:
            # Both are orth OR Psi is orth, Phi is not orth
            if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                    is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                # Y=(2*PsiTX*B'+rho_1*Z+Gamma_1)*inv(2*(B*B')+rho_1*I_y+exp(-15));
                Y = (2 * PsiTX.astype(np.complex128) @ (B.astype(np.complex128)).conj().T + rho_1 * Z.astype(
                    np.complex128) + gamma_1) \
                    @ np.linalg.pinv(2 * (B @ B.conj().T) + rho_1 * I_Y + math.exp(-15)).astype(np.complex128)
            elif (is_orthonormal(phi_d) and not is_orthonormal(psi_d)) or (
                    not is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
                Y = hard_update_y(sigma, W, D, psi_d, phi_d, lam_1, Q_1, gamma_1, rho_1)

        test_instance = TestYConversion()
        ans_y = test_instance.test_y_complex_conversion(i, Y)
        # Update Z:
        # h = Y-gamma_1 / rho_1
        # Z = sign(h).*max(abs(h)-lambda_1/rho_1, 0)
        h = (Y - (gamma_1.astype(np.complex128) / rho_1)).astype(np.complex128)
        Z = (np.sign(h) * np.maximum(np.abs(h) - (lambda_1 / rho_1), 0)).astype(np.complex128)
        # A = psi*Y*sigma
        A = psi_d.astype(np.longdouble) @ Y.astype(np.complex128) @ sigma.astype(np.longdouble)
        if mask.any():
            # Both are orthonormal OR Psi is orth, Phi is not
            if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                    is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
                # W = inv(2*(A')*A + I_W*rho_2) * (2*A'*D*phi'+rho_2*V+gamma_2)
                W = np.linalg.pinv(2 * A.conj().T @ A + I_W * rho_2).astype(np.complex128) @ (
                        2 * A.conj().T @ D.astype(np.complex128) @ phi_d.conj().T + rho_2 * V + gamma_2)
            # Psi is orth, Phi is not orth OR Psi is not orth, Phi is not orth
            elif (is_orthonormal(psi_d) and not is_orthonormal(phi_d)) or (
                    not is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                W = hard_update_w(sigma, V, D, Y, psi_d.astype(np.longdouble), phi_d.astype(np.complex128), lam_4, Q_4,
                                  gamma_2, rho_2)
        else:
            # Both are orth OR Phi is orth, Psi is not
            if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                    is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
                # W=inv(2*(A')*A + I_w*rho_2)*(2*A'*XPhiT+rho_2*V+ Gamma_2);
                W = np.linalg.pinv(2 * A.conj().T @ A + I_W * rho_2).astype(np.complex128) @ (
                        2 * A.conj().T @ XPhiT.astype(np.complex128) + rho_2 * V + gamma_2)
            # Psi is orth, Phi is not
            elif is_orthonormal(psi_d) and not is_orthonormal(phi_d):
                sigma = sigma @ sigmaR
                W = hard_update_w(sigma, V, D, Y, psi_d.astype(np.longdouble), phi_d.astype(np.complex128), lam_4, Q_4,
                                  gamma_2, rho_2)
            elif not is_orthonormal(psi_d) and not is_orthonormal(phi_d):
                W = hard_update_w(sigma, V, D, Y, psi_d.astype(np.longdouble), phi_d.astype(np.complex128), lam_4, Q_4,
                                  gamma_2, rho_2)

        test_instance_w = TestWConversion()
        ans_w = test_instance_w.test_w_complex_conversion(i, W)
        # Update V:
        # h= W-Gamma_2/rho_2;
        # V = sign(h).*max(abs(h)-lambda_2/rho_2,0);
        h = W - (gamma_2.astype(np.complex128) / rho_2)
        V = (np.sign(h) * np.maximum(np.abs(h) - (lambda_2 / rho_2), 0))

        gamma_1, gamma_2 = gamma_1 + rho_1 * (Z - Y), gamma_2 + rho_2 * (V - W)
        rho_1, rho_2 = min(rho_1 * 1.1, 1e5), min(rho_2 * 1.1, 1e5)

        # Stop condition
        if i % 25 == 0:
            obj_new = get_object(mask, D, X, phi_d, psi_d, Y, sigma, W, lambda_1, lambda_2, lambda_3)
            objs = [objs, obj_new]
            residual = abs(obj_old - obj_new)
            print(f"obj-{i}={obj_new}, residual-{i}={residual}")
            if residual < 1e-6:
                break
            else:
                obj_old = obj_new
    # plt.xlabel('Iteration')
    # plt.ylabel('Values')
    # plt.title('Convergence of Y and W')
    # plt.grid(True)
    # Display the final plot
    # plt.show()

    return Y, W


def find_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_percentage, p_count) -> None:
    """
    Plots outliers based on magnitude from the residual of X-(Ψ * p_Y * p_W * p_Φ).
    Args:
        p_X: Original temporal signal input
        p_Psi: Graph dictionary Ψ
        p_Y: Encoding matrix to reconstruct p_X
        p_W: Encoding matrix to reconstruct p_X
        p_Phi: Time series p_Φ
        p_percentage: Top percentile of outliers that the user wishes to plot
        p_count: Number of subplots to display, one for each outlier in p_count. Maximum count of 10.

    """
    # Compute the residual and the reconstructed X
    res = p_X - (p_Psi @ p_Y @ p_W @ p_Phi)
    reconstructed_X = p_Psi @ p_Y @ p_W @ p_Phi

    # Used for indexing
    flatten_residual = res.flatten()
    sorted_values = np.argsort(np.abs(flatten_residual))[::-1]

    # Find the percentage of how many outliers there will be based on input
    num_outliers_percentage = int(len(flatten_residual) * p_percentage / 100)
    outlier_indices_percentage = sorted_values[:num_outliers_percentage]
    row_indices_percentage, col_indices_percentage = np.unravel_index(outlier_indices_percentage, p_X.shape)

    # Determine the indices of the top fixed number of outliers
    num_outliers_for_subplots = min(p_count, 10)
    outlier_indices_fixed = sorted_values[:num_outliers_for_subplots]
    row_indices_fixed, col_indices_fixed = np.unravel_index(outlier_indices_fixed, p_X.shape)

    # Use magnitude for X and reconstructed X
    X_magnitude = np.abs(p_X)
    reconstructed_X = np.abs(reconstructed_X)

    # Plotting!
    fig = plt.figure(figsize=(18, 4 * num_outliers_for_subplots))
    # 3 "grids"
    gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 1, 1])

    # Subplot 1: Original Data with Percentage-Based Outliers
    ax_matrix = fig.add_subplot(gs[0])
    ax_matrix.imshow(X_magnitude, cmap='gray')
    ax_matrix.scatter(col_indices_percentage, row_indices_percentage, color='red', s=50,
                      label=f'Top {p_percentage}% Outliers')
    ax_matrix.set_xlabel('Column Index')
    ax_matrix.set_ylabel('Row Index')
    ax_matrix.set_title('Original Data X with Top % Outliers')

    # Subplot 2: Local Neighborhood of Outliers Corresponding to the Time Series
    # Groups the outliers (row, col) by row index so that they can be plotted on the same subplot together
    outliers_by_series = defaultdict(list)
    for row_idx, col_idx in zip(row_indices_fixed, col_indices_fixed):
        Phi_row_idx = row_idx % p_Phi.shape[0]
        outliers_by_series[Phi_row_idx].append((row_idx, col_idx))

    # Left hand side will have the local neighborhood of the outliers with the time series
    left_column_rhs = GridSpecFromSubplotSpec(min(len(outlier_indices_fixed), 10), 1, subplot_spec=gs[1])
    # Right hand side will have the fit
    right_column_rhs = GridSpecFromSubplotSpec(10, 1, subplot_spec=gs[2])  # Assuming 10 new subplots

    time_points = np.arange(p_Phi.shape[1])
    for i, (phi_row_idx, indices) in enumerate(outliers_by_series.items()):
        # Add vertical subplots down each column
        ax_ts = fig.add_subplot(left_column_rhs[i])
        ax_ts_right = fig.add_subplot(right_column_rhs[i])
        # Fetches the time series at that particular index (by row)
        time_series = p_Phi[phi_row_idx, :]

        for row_idx, col_idx in indices:
            # Finds the local neighborhood
            start = max(col_idx - num_outliers_for_subplots, 0)
            end = min(col_idx + num_outliers_for_subplots + 1, len(time_series))

            # Plots the local neighborhood (time series) along with the corresponding X
            ax_ts.plot(time_points[start:end], time_series[start:end], label='Local Neighborhood' if i == 0 else "")
            ax_ts.scatter(time_points[col_idx], time_series[col_idx], color='red', zorder=5,
                          label='Outlier' if i == 0 else "")
            ax_ts.annotate(f'({row_idx}, {col_idx})',
                           (time_points[col_idx], time_series[col_idx]),
                           textcoords="axes fraction",
                           xytext=(0.95, 0.05),
                           ha='right',
                           va='bottom',
                           fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.5),
                           fontsize=8)

            # Subplot 3: Fit of the Outlier Corresponding to X in Local Neighborhood
            # For the same neighborhood, plot X values against reconstructed values
            ax_ts_right.plot(time_points[start:end], X_magnitude[row_idx, start:end], 'b-', label='Local Neighborhood of X' if i == 0 else "")
            ax_ts_right.plot(time_points[start:end], reconstructed_X[row_idx, start:end], 'g--', label='Reconstructed Neighborhood of X' if i == 0 else "")
            ax_ts_right.scatter(col_idx, X_magnitude[row_idx, col_idx], color='blue', zorder=5, label='Actual Outlier (X)' if i == 0 else "")
            ax_ts_right.scatter(col_idx, reconstructed_X[row_idx, col_idx], color='green', zorder=5, label='Reconstructed Outlier (X)' if i == 0 else "")

            # Sets the x-axis for each subplot
            ax_ts.set_xlim(start, end)
            ax_ts_right.set_xlim(start, end)

            # Right hand side: legend, since no specific point is necessary but just the representation
            if i == 0:
                handles, labels = ax_ts_right.get_legend_handles_labels()
                ax_ts_right.legend(handles, labels, loc='upper right', bbox_to_anchor=(2.15, 1.3), fontsize='small')

        # Sets y-axis for each subplot
        y_min, y_max = time_series[start:end].min(), time_series[start:end].max()
        ax_ts.set_ylim(y_min, y_max)
        y_min_right, y_max_right = np.min(X_magnitude), np.max(X_magnitude)
        ax_ts_right.set_ylim(y_min_right, y_max_right)

        # For the last (bottom) subplot only
        if i == num_outliers_for_subplots - 1:
            ax_ts.set_xlabel('Time Index')
            ax_ts_right.set_xlabel('Time Index')

        ax_ts.grid(True)
        ax_ts_right.grid(True)

    # Adjust as needed for visualization
    plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
    plt.show()

def find_row_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count):
    """
    Plots row outliers based on average magnitude from the residual of X-(Ψ * p_Y * p_W * p_Φ).
    Args:
        p_X: Original temporal signal input
        p_Psi: Graph dictionary Ψ
        p_Y: Encoding matrix to reconstruct p_X
        p_W: Encoding matrix to reconstruct p_X
        p_Phi: Time series p_Φ
        p_count: Number of subplots to display, one for each row outlier in p_count. Maximum count of 10.
    """
    # Calculate the residual and reconstruction of X
    res = p_X - (p_Psi @ p_Y @ p_W @ p_Phi)
    reconstructed_X = (p_Psi @ p_Y @ p_W @ p_Phi)
    magnitude_X = abs(p_X)
    reconstructed_X = abs(reconstructed_X)

    # Take the average of each row
    row_avg = np.abs(res).mean(axis=1)

    # Determine the number of rows to plot based on user input and sort average value
    sorted_rows = np.argsort(row_avg)[::-1]
    p_count = min(p_count, 10)
    outlier_rows = sorted_rows[:p_count]
    num_plots = len(outlier_rows)

    # Define grid
    fig = plt.figure(figsize=(15, 3 * p_count))
    gs = GridSpec(num_plots, 2)  # Define grid layout for the figure

    # Iterate through each row index to build grid
    for i, row_idx in enumerate(outlier_rows):
        # Subplot 1: Average Value of the Row Plotted Against the Time Series
        ax = fig.add_subplot(gs[i, 0])

        avg_value = np.mean(p_X[row_idx, :])  # Average value for the outlier row in X
        time_series = p_Phi[row_idx % p_Phi.shape[0], :]  # Corresponding time series in Phi
        differences = np.abs(time_series - avg_value)
        closest_index = np.argmin(differences) # Index of the minimum difference (in other words, the closest time point for this avg. value)

        ax.plot(time_series, color='blue')
        # Highlight the point closest to the average value
        ax.scatter(closest_index, time_series[closest_index], color='red', zorder=5)

        # Annotate the row index
        ax.annotate(f'Row {row_idx}', xy=(0.0, 0.95), xycoords='axes fraction',
                        ha='left', va='top',
                        fontweight='bold', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.5))

        # Set y-axis
        y_min, y_max = time_series.min(), time_series.max()
        ax.set_ylim(y_min, y_max)

        # Subplot 2: Plot the Row Values Against Their Reconstructed Values (Fit)
        ax_compare = fig.add_subplot(gs[i, 1])
        ax_compare.plot(magnitude_X[row_idx, :], 'b-', label='X')
        ax_compare.plot(reconstructed_X[row_idx, :], 'g--', label='Reconstructed X')

        # Set y-axis
        ax_compare.set_ylim(np.min(magnitude_X), np.max(magnitude_X))

        # Add legend to first subplot
        if i == 0:
            handles, labels = ax_compare.get_legend_handles_labels()
            ax_compare.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.3), fontsize='small')

        # Enable the xlabel only for bottom subplot
        if i == num_plots - 1:
            ax.set_xlabel('Time Index')
            ax_compare.set_xlabel('Time Index')
        else:
            ax.tick_params(labelbottom=False)
            ax_compare.tick_params(labelbottom=False)

        ax.grid(True)
        ax_compare.grid(True)

    # Adjust as needed for visualization
    plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
    plt.show()


def find_col_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count):
    """
    Plots column outliers based on average magnitude from the residual of X-(Ψ * p_Y * p_W * p_Φ).
    Args:
        p_X: Original temporal signal input
        p_Psi: Graph dictionary Ψ
        p_Y: Encoding matrix to reconstruct p_X
        p_W: Encoding matrix to reconstruct p_X
        p_Phi: Time series p_Φ
        p_count: Number of subplots to display, one for each column outlier in p_count. Maximum count of 10.
    """
    # Calculate the residual and reconstruction of X
    res = p_X - (p_Psi @ p_Y @ p_W @ p_Phi)
    magnitude_X = abs(p_X)
    reconstructed_X = p_Psi @ p_Y @ p_W @ p_Phi
    reconstructed_X = abs(reconstructed_X)

    # Take the average of each column
    col_avg = np.abs(res).mean(axis=0)
    sorted_columns = np.argsort(col_avg)[::-1]
    outlier_columns = sorted_columns[:p_count]

    # Determine number of time series to plot
    num_series_to_plot = min(p_Phi.shape[0], 10)  # Plot up to the first 10 time series
    fig = plt.figure(figsize=(15, 3 * p_count))
    gs = GridSpec(num_series_to_plot, 2)  # Define grid layout for the figure

    # Iterate through each column outlier index
    for i, col_idx in enumerate(outlier_columns):
        # Subplot 1: Average Value of the Column Plotted Against the Time Series
        ax = fig.add_subplot(gs[i, 0])
        # Extracting the vertical time series for the outlier column
        vertical_series = p_Phi[:, col_idx]
        ax.plot(vertical_series, color='blue')

        # Highlighting the entire column as an outlier
        ax.axvline(x=col_idx, color='red', linestyle='--', label=f'Outlier at {col_idx}')

        # Plot the column index
        ax.annotate(f'Column {col_idx}', xy=(0.0, 0.95), xycoords='axes fraction', ha='left', va='top',
                        fontweight='bold', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.5))
        ax.invert_yaxis()  # Invert y-axis to have the top of the plot as the start of the series

        # Set y-axis
        y_min, y_max = vertical_series.min(), vertical_series.max()
        ax.set_ylim(y_min, y_max)

        # Subplot 2: Plot the Column Values Against Their Reconstructed Values (Fit)
        ax_compare = fig.add_subplot(gs[i, 1])
        ax_compare.plot(magnitude_X[:, col_idx], 'b-', label='X')
        ax_compare.plot(reconstructed_X[:, col_idx], 'g--', label='Reconstructed X')

        # Set y-axis
        ax_compare.set_ylim(np.min(magnitude_X), np.max(magnitude_X))

        # Add legend to first subplot
        if i == 0:
            handles, labels = ax_compare.get_legend_handles_labels()
            ax_compare.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.3), fontsize='small')

        # Enable the xlabel only for bottom subplot
        if i == num_series_to_plot - 1:
            ax.set_xlabel('Time Index')
            ax_compare.set_xlabel('Time Index')
        else:
            ax.tick_params(labelbottom=False)
            ax_compare.tick_params(labelbottom=False)

        ax.grid(True)
        ax_compare.grid(True)

    # Adjust as needed for visualization
    plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
    plt.show()



###################################################################################################
def mdtm(is_syn: bool, X, adj, mask, count_nnz=10, num_iters_check=0, lam=0.000001, K=10, epsilon=1e-4):
    def gen_syn_X(p_syn):
        # Create a list of matrices from the cell array
        return tl.kruskal_to_tensor((np.ones(p_syn['Kgen'][0, 0]), [matrix for matrix in p_syn['PhiYg'][0]]))

    def gen_syn_lambda_rho(p_syn):
        syn_lambda = [0.000001 for _ in p_syn['dimension'][0, ]]
        syn_rho = [val * 5 for val in syn_lambda]
        return syn_lambda, syn_rho

    def mttkrp(p_D, p_PhiY, p_n):
        return tl.unfold(p_D, mode=p_n) @ tl.tenalg.khatri_rao([p_PhiY[i] for i in range(len(p_PhiY)) if i != p_n])

    def nd_index(index, shape):
        indices = [0] * len(shape)
        for i in range(len(shape) - 1, -1, -1):
            stride = np.prod(shape[:i])
            indices[i], index = divmod(index, stride)
            indices[i] = int(indices[i])
        return tuple(indices)

    if is_syn:
        s_data = load_syn_data()
        X = gen_syn_X(s_data)  # numpy array of x * y * z
        num_modes = X.ndim
        lam, rho = gen_syn_lambda_rho(s_data)  # list of size n
        phi_d = s_data['Phi']  # list of numpy arrays in form (1, n) where each atom corresponds to a dictionary.
        phi_type = ['not_ortho_dic', 'ortho_dic', 'no_dic']
        # first coordinate of each dictionary = shape of X
        P = s_data['P']  # Y values of shape of X
    else:
        if len(X) == 0:
            raise Exception
        else:
            num_modes = X.ndim
            temp = lam
            lam = [temp] * num_modes if temp is not None else 0.000001
            rho = [val * 5 for val in lam]
            phi_d = []
            for mode in range(num_modes):
                # Fill with dictionaries
                #if mode == 0 or mode == 1:
                    #nested_dictionary = gen_gft(p_dict, False) # Fix adjacency matrix here
                #phi_d.append(nested_dictionary)
                return

            # Convert the list to a numpy object array of size (1, N)
            phi_d = np.empty((1, num_modes), dtype=object)
            for i in range(num_modes):
                phi_d[0, i] = phi_d[i]

            phi_type = [random.choice(['not_ortho_dic', 'ortho_dic', 'no_dic']) for _ in range(num_modes)]
            P = []
            for i in len(phi_d):
                y_idx = phi_d[i].shape[1]
                if y_idx > 250:
                    dtype = np.uint16
                elif y_idx > 100:
                    dtype = np.uint8
                else:
                    dtype = np.uint8
                    new_array = np.array([[y_idx]], dtype=dtype)
                    P.append(new_array)
            P = np.array([P], dtype=object)

    # In the case of no dictionary, edit P manually
    for _ in range(len(phi_type)):
        num = X.shape[_]
        if phi_type[_] == "no_dic":
            P[0, _] = np.array([[num]], dtype=np.uint16)

    # this ignores the try statements at initialization in MDTM.m
    # cast to Double for mask indexing i.e. double_X = double(X)
    D = X
    normalize_scaling = np.ones(K)
    dimensions = X.shape
    mask_complex = 1
    set_difference = None if not mask_complex else set(range(1, dimensions[0] * dimensions[1] * dimensions[2] + 1)) - set(mask)
    mask_complex = 1 if not mask_complex else mask_complex
    if mask and mask_complex == 1:
        double_X = X.astype(np.longdouble)
    else:
        mask_i, mask_j, mask_t = np.unravel_index(np.array(mask, dtype=np.intp), dimensions)
        # mask_tensor = sptensor([mask_i', mask_j', mask_t'], 1);
        # stack indices and create coordinate list sparse tensor, convert to tensor format
        mask_tensor = sp_tensor(sparse.COO(np.vstack((mask_i, mask_j, mask_t)), np.ones_like(mask_i), shape=dimensions),
                                dtype='float')

    np.random.seed(6)

    PhiPhiEV, PhiPhiLAM = [None] * num_modes, [None] * num_modes
    # dictionary decomposition

    for i in range(num_modes):
        if phi_type[i] == "not_ortho_dic":
            # U and V are unitary matrices and S contains singular values as a diagonal matrix
            U, S_non_diag, V_transpose = np.linalg.svd(phi_d[0, i])
            S = np.diag(S_non_diag)
            V = V_transpose.T
            rows_v, cols_v = V.shape
            PhiPhiEV[i] = V
            retained_rank = min(phi_d[0, i].shape)
            # diag(S(1:retained_rank, 1:retained_rank)).^2
            PhiPhiLAM[i] = (np.diag(S[:retained_rank]) ** 2).reshape(-1, 1)

    # 1:num_modes, @(x) isequal(sort(x), 1:num_modes)
    MAX_ITERS = 500
    # 1:num_modes, @(x) isequal(sort(x), 1:num_modes)
    dimorder_check = lambda x: sorted(x) == list(range(1, num_modes + 1))
    dimorder = np.arange(1, num_modes + 1)
    gamma = [None] * num_modes

    YPhiInitInner = np.zeros((K, K, num_modes))
    Y_init, PhiYInit = [None] * num_modes, [None] * num_modes

    for n in range(len(dimorder)):
        Y_init[n] = np.random.rand(P[0, n][0, 0], K)
        PhiYInit[n] = (phi_d[0, n] @ Y_init[n]) if phi_type[n] in ["not_ortho_dic", "ortho_dic"] else Y_init[n]
        YPhiInitInner[:, :, n] = PhiYInit[n].T @ PhiYInit[n] if phi_type[n] == "not_ortho_dic" else Y_init[n].T @ \
                                                                                                    Y_init[n]
        gamma[n] = 0

    # set up for initialization, U and the fit
    Y = Y_init
    PhiY = PhiYInit
    YPhi_Inner = YPhiInitInner
    Z = Y
    fit = 0
    recon_t = tl.kruskal_to_tensor((normalize_scaling, [matrix for matrix in PhiY]))
    normX = tl.norm(tl.tensor(X), order=2)
    objs = np.zeros((MAX_ITERS, num_modes))
    objs[0, 0] = tl.sqrt((normX ** 2) + (tl.norm(recon_t, order=2) ** 2) - 2 * tl.tenalg.inner(X, recon_t))
    # iterate until convergence
    avg_time = 0
    for i in range(1, MAX_ITERS + 1):
        tic = time.time()  # start time
        for n in range(len(dimorder)):
            if phi_type[n] == "not_ortho_dic":
                # calculate Unew = Phi X_(n) * KhatriRao(all U except n, 'r')
                product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)
                pv, Ev = np.linalg.eigh(product_vector)
                pv = pv.reshape(-1, 1)

                CC = PhiPhiEV[n].T @ (phi_d[0, n].T @ mttkrp(D, PhiY, n) + rho[n] * Z[n] - gamma[n]) @ Ev
                Y[n] = CC / (rho[n] + PhiPhiLAM[n] @ pv.T)
                Y[n] = PhiPhiEV[n] @ Y[n] @ Ev.T
                PhiY[n] = phi_d[0, n] @ Y[n]
                # normalize_scaling = sqrt(sum(Y{n}.^2, 1))' else max(max(abs(Y{n}), [], 1), 1)'
                normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1, 1).T if i == 1 else np.maximum(
                    np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                Y[n] /= normalize_scaling
                PhiY[n] = phi_d[0, n] @ Y[n]
                YPhi_Inner[:, :, n] = PhiY[n].T @ PhiY[n]

            elif phi_type[n] == "ortho_dic":
                # phi_d_rao_other_factors = (phi_d[0, n].T @ mttkrp(D, PhiY, n) + rho[n] * Z[n] - gamma[n])
                product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)
                # denominator = product_vector + rho[n] * np.eye(K)
                Y[n] = np.linalg.solve((product_vector + rho[n] * np.eye(K)).T,
                                       (phi_d[0, n].T @ mttkrp(D, PhiY, n) + rho[n] * Z[n] - gamma[n]).T).T
                normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1, 1).T if i == 1 else np.maximum(
                    np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                Y[n] /= normalize_scaling
                PhiY[n] = phi_d[0, n] @ Y[n]
                YPhi_Inner[:, :, n] = Y[n].T @ Y[n]

            elif phi_type[n] == "no_dic":
                Y[n] = mttkrp(D, PhiY, n)
                # inversion_product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)
                Y[n] = np.linalg.solve(
                    (np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)).T, Y[n].T).T
                normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1, 1).T if i == 1 else np.maximum(
                    np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                Y[n] /= normalize_scaling
                PhiY[n] = Y[n]
                YPhi_Inner[:, :, n] = Y[n].T @ Y[n]
            else:
                return

            h = Y[n] - gamma[n] / rho[n]
            Z[n] = np.sign(h) * np.maximum(np.abs(h) - (lam[n] / rho[n]), 0)
            gamma[n] = gamma[n] + rho[n] * (Z[n] - Y[n])
            if count_nnz != 0:
                nnz = 0
                for m in range(len(dimensions)):
                    # nnz = nnz + length(find(Z{m} ~= 0)
                    nnz = nnz + np.count_nonzero(Z[m])
                objs[i - 1, 2] = nnz

        if len(mask) > 0:
            mask = np.array(mask)
            # set D to reconstructed values and cast to double for mask indexing
            recon_t = tl.kruskal_to_tensor(
                (normalize_scaling.reshape((normalize_scaling.shape[1],)), [matrix for matrix in PhiY]))
            D = recon_t

            missing_mask, observed_mask = np.zeros(double_X.shape), np.ones(double_X.shape)

            for idx in mask:
                nd_idx = nd_index(idx, double_X.shape)  # Convert index to correct tuple
                missing_mask[nd_idx] = 1

            # d = (recon_t(:) + lambda * X(:)) * inv(1 + lambda)
            # D(setDifference) = d

            D = ((observed_mask - missing_mask) * (recon_t + lam[0] * double_X)) / 1 + lam[0]
        else:
            D = X

        time_one_iter = time.time()
        total_time_one_iter = time_one_iter - tic
        avg_time += total_time_one_iter
        if i % num_iters_check == 0:
            sparsity_constraint = 0
            for n in range(len(dimorder)):
                sparsity_constraint = sparsity_constraint + lam[n] * np.sum(np.abs(Y[n]))

            if len(mask) == 0 or mask_complex == 1:
                recon_t = tl.kruskal_to_tensor((np.squeeze(normalize_scaling), [matrix for matrix in PhiY]))

            recon_error = tl.sqrt((normX ** 2) + (tl.norm(recon_t, order=2) ** 2) - 2 * tl.tenalg.inner(X, recon_t))

            fit = 1 - (recon_error / normX)

            objs[i // num_iters_check, 0] = fit  # 2, 1 == 1, 0
            objs[i // num_iters_check, 1] = objs[i // num_iters_check - 1, 1] + total_time_one_iter  # 2, 2 and 1, 2
            fit_change = np.abs(objs[i // num_iters_check, 0] - objs[i // num_iters_check - 1, 0])

            print(
                f"Iteration={i} Fit={fit} f-delta={fit_change} Reconstruction Error={recon_error} Sparsity Constraint={sparsity_constraint} Total={objs[i // num_iters_check, :]}")

            if fit_change < epsilon:
                print(f"{avg_time}, {i}")
                print(f"Algo has met fit change tolerance, avg time: {avg_time / i}")
                break

    # [[S ⊡ Φ1Y1, Φ2Y2, Φ3Y3]
    return X, recon_t


def mdtm_load_config():
    with open('mdtm_config.json', 'r') as file:
        config = json.load(file)

    # Load X from a CSV file
    X = pd.read_csv(config['X']).values if config['X'] is not None else None

    if X is not None and config['adj']:
        # TODO
        adj = config['adj']
    if X is not None and config['mask_percentage_random']:
        num_elements_to_mask = int(np.prod(X.shape) * (config['mask_percentage_random'] / 100.0))
        mask = np.random.choice(np.prod(X.shape), size=num_elements_to_mask, replace=False)
    else:
        mask = None

    # The rest of the configuration parameters
    count_nnz: config['count_nnz']
    num_iters_check: config['num_iters_check']
    lam = config['lam']
    K = config['K']
    epsilon = config['epsilon']

    return X, adj, mask, count_nnz, num_iters_check, lam, K, epsilon


def mdtm_find_outlier(p_X, p_recon, p_count) -> None:
    """
    Plots outliers based on magnitude from the residual of p_X-(p_recon).
    Args:
        p_X: Original temporal signal tensor
        p_recon: Reconstruction of p_X from S ⊡ Φ1Y1, Φ2Y2, Φ3Y3
        p_percentage: Top percentile of outliers that the user wishes to plot
        p_count: Number of subplots to display, one for each outlier in p_count. Maximum count of 10.
    """
    #TODO
    outlier_indices = []
    p_X, p_recon = np.abs(p_X), np.abs(p_recon)
    residual = p_X-p_recon
    p_count = min(p_count, 10)

    for i in range(p_X.shape[2]):
        slice_flattened = residual[:, :, i].flatten()
        sorted_values = np.argsort(slice_flattened)[::-1]

        # Determine the indices of the top fixed number of outliers
        outlier_indices_fixed = sorted_values[:p_count]
        row_indices_fixed, col_indices_fixed = np.unravel_index(outlier_indices_fixed, p_X[:, :, i].shape)
        for row, col in zip(row_indices_fixed, col_indices_fixed):
            outlier_indices.append((row, col, i))

    pane_max_outlier_magnitude = {}
    for (row, col, pane) in outlier_indices:
        # Calculate the magnitude of the outlier - replace this with your actual calculation if different
        magnitude = p_X[row, col, pane]

        # Update the maximum outlier magnitude for each pane
        if pane in pane_max_outlier_magnitude:
            pane_max_outlier_magnitude[pane] = max(pane_max_outlier_magnitude[pane], magnitude)
        else:
            pane_max_outlier_magnitude[pane] = magnitude

    top_panes = sorted(pane_max_outlier_magnitude, key=pane_max_outlier_magnitude.get, reverse=True)[:25]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for pane_idx in top_panes:
        pane_outliers = [(row, col, z) for row, col, z in outlier_indices if z == pane_idx]
        magnitudes = [p_X[row, col, z] for row, col, z in pane_outliers]
        sizes = [(magnitude / max(magnitudes)) * 100 for magnitude in magnitudes]
        rows, cols, zs = zip(*pane_outliers)
        ax.scatter(rows, cols, zs=pane_idx, s=sizes, depthshade=True, label=f'Pane {pane_idx}')

    # Set labels and title
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel('Row Index')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylabel('Column Index')

    ax.set_zlabel('Pane Index')
    ax.set_title('Top 25 Panes with 10 Largest Outliers')

    # Set the ticks for the z-axis (pane index) to correspond to the indices of the panes
    ax.set_xticks([0, p_X.shape[0]])
    ax.set_yticks([0, p_X.shape[1]])
    ax.set_zticks([min(top_panes), max(top_panes)])

    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=120)  # Elevate and rotate the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.10, 0.5), title='Pane Number')

    plt.show()
def mdtm_find_row_outlier(p_X, p_recon, p_count) -> None:
    """
    Plots row outliers based on average row magnitude from the residual of p_X-(p_recon).
    Args:
        p_X: Original temporal signal tensor
        p_recon: Reconstruction of p_X from S ⊡ Φ1Y1, Φ2Y2, Φ3Y3
        p_count: Number of subplots to display, one for each row outlier in p_count. Maximum count of 10.
    """
    #TODO

def mdtm_find_col_outlier(p_X, p_recon, p_count) -> None:
    """
    Plots column outliers based on average column magnitude from the residual of p_X-(p_recon).
    Args:
        p_X: Original temporal signal tensor
        p_recon: Reconstruction of p_X from S ⊡ Φ1Y1, Φ2Y2, Φ3Y3
        p_count: Number of subplots to display, one for each column outlier in p_count. Maximum count of 10.
    """
    #TODO

###################################################################################################

mat = load_matrix()
# ram = gen_rama(400, 10)
#mdtm_input_x, mdtm_input_adj, mdtm_input_mask, mdtm_input_count_nnz, mdtm_input_num_iters_check, mdtm_input_lam, mdtm_input_K, mdtm_input_epsilon = mdtm_load_config()
mdtm_X, recon_X = mdtm(is_syn=True, X=None, adj=None, mask=[], count_nnz=0, num_iters_check=10, lam=0.000001, K=10,
                       epsilon=1e-4)
mdtm_find_outlier(mdtm_X, recon_X, 10)
Psi_GFT = gen_gft_new(mat['adj'], False)
Psi_GFT = Psi_GFT[0]  # eigenvectors
Phi_DFT = gen_dft(200)
# non_orth_psi = Psi_GFT + 0.1 * np.outer(Psi_GFT[:, 0], Psi_GFT[:, 1])
# non_orth_phi = Phi_DFT + 0.1 * np.outer(Phi_DFT[:, 0], Phi_DFT[:, 1])

Y, W = tgsd(mat['X'], Psi_GFT, Phi_DFT, mat['mask'], iterations=100, k=7, lambda_1=.1, lambda_2=.1, lambda_3=1,
            rho_1=.01, rho_2=.01, type="rand")

pred_matrix = Psi_GFT @ Y @ W @ Phi_DFT
#find_outlier(mat['X'], Psi_GFT, Y, W, Phi_DFT, .1, 25)
#find_row_outlier(mat['X'], Psi_GFT, Y, W, Phi_DFT, 10)
#find_col_outlier(mat['X'], Psi_GFT, Y, W, Phi_DFT, 10)
print(pred_matrix)
