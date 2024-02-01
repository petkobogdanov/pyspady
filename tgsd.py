import json
import os
import random
import cmath
import itertools
import math
from math import gcd, pi
import numpy as np
import scipy.io
import tensorly as tl
import sparse
from tensorly.contrib.sparse import tensor as sp_tensor
import scipy.sparse as sp
from scipy.fftpack import fft
import time
import matplotlib.pyplot as plt
from Y_unittest import TestYConversion
from W_unittest import TestWConversion

from numba import njit, prange, jit
from numba.typed import List

def load_matrix(file: str) -> dict:
    """
    Loads matrix from file in directory
    Returns:
        dict: dictionary with variable names as keys, and loaded matrices as values
    """
    return scipy.io.loadmat(file)


def load_matrix_demo() -> dict:
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
        if type == "row": # row-mask
            n, m = X.shape
            temp2 = np.ones((n, m))
            temp2[(mask-1) % X[0], :] = 0
            mask = np.argwhere(temp2 == 0)
        elif type == "col" or type == "pred": # column or pred mask
            n, m = X.shape
            temp2 = np.ones((n, m))
            temp2[:, (mask-1) // X[0]] = 0
            mask = np.argwhere(temp2 == 0)
        # else "rand" by default

    #plt.figure()

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
        # plt.plot(i, ans_y, 'ro', markersize=5)
        # plt.plot(i, ans_w, 'bo', markersize=5)
        # plt.pause(0.1)  # Adjust the pause duration as needed
        print("---")
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

@njit
def unfold_fast(tensor, mode):
    I, J, K = tensor.shape
    if mode == 0:
        # Mode-0 unfolding (I, J * K)
        unfolded = np.empty((I, J * K))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    unfolded[i, j * K + k] = tensor[i, j, k]
        return unfolded
    elif mode == 1:
        # Mode-1 unfolding (J, I * K)
        unfolded = np.empty((J, I * K))
        for j in range(J):
            for i in range(I):
                for k in range(K):
                    unfolded[j, i * K + k] = tensor[i, j, k]
        return unfolded
    elif mode == 2:
        # Mode-2 unfolding (K, I * J)
        unfolded = np.empty((K, I * J))
        for k in range(K):
            for i in range(I):
                for j in range(J):
                    unfolded[k, i * J + j] = tensor[i, j, k]
        return unfolded

@njit(parallel=True, cache=True)
def khatri_rao(matrices):
    # Check if matrices are 2D and have the same number of columns
    n_columns = matrices[0].shape[1]
    # Compute the number of rows in the result
    n_rows = 1
    for m in matrices:
        n_rows *= m.shape[0]

    # Initialize the result matrix
    result = np.ones((n_rows, n_columns))

    for col in prange(n_columns):
        # For each column, compute the Kronecker product
        for i in prange(matrices[0].shape[0]):
            for j in prange(matrices[1].shape[0]):
                # The row index in the result is computed based on the row indices in each matrix
                result_row_index = i * matrices[1].shape[0] + j
                # Compute the product for the current column
                result[result_row_index, col] = matrices[0][i, col] * matrices[1][j, col]

    return result

def mdtm(is_syn, X, mask, phi_type, phi_d, P, lam, rho, K, epsilon, num_modes, count_nnz=None, num_iters_check=None,
         mask_complex=None):
    def gen_syn_X(p_syn):
        # Create a list of matrices from the cell array
        return tl.kruskal_to_tensor((np.ones(p_syn['Kgen'][0, 0]), [matrix for matrix in p_syn['PhiYg'][0]]))

    def gen_syn_lambda_rho(p_syn):
        syn_lambda = [0.000001 for _ in p_syn['dimension'][0,]]
        syn_rho = [val * 5 for val in syn_lambda]
        return syn_lambda, syn_rho

    def mttkrp(p_D, p_PhiY, p_n):
        return tl.unfold(p_D, mode=p_n) @ tl.tenalg.khatri_rao([p_PhiY[i] for i in range(len(p_PhiY)) if i != p_n])

    def updated_mttkrp(p_D, p_PhiY, p_n):
        foo = [p_PhiY[i] for i in range(len(p_PhiY)) if i != n]
        typed_foo = List()
        for mat in foo:
            typed_foo.append(np.ascontiguousarray(mat))
        return unfold_fast @ khatri_rao(typed_foo)


    if is_syn:
        s_data = load_syn_data()
        X = gen_syn_X(s_data)
        lam, rho = gen_syn_lambda_rho(s_data)
        phi_type = ["ortho_dic", "ortho_dic", "not_ortho_dic"]
        phi_d = s_data['Phi']
        P = s_data['P']

    #for i in range(1):
    #    phi_d[0, i] = np.array([])
    #    phi_type[i] = "no_dic"
    #    P[0, i] = np.array([[X.shape[i]]])
    # this ignores the try statements at initialization in MDTM.m
    # cast to Double for mask indexing i.e. double_X = double(X)
    D = X
    normalize_scaling = np.ones(K)
    dimensions = X.shape
    count_nnz = 0 if not count_nnz else count_nnz
    num_iters_check = 10 if not num_iters_check else num_iters_check
    set_difference = None if not mask_complex else set(
        range(1, dimensions[0] * dimensions[1] * dimensions[2] + 1)) - set(mask)
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

        if mask:
            if mask_complex == 1:
                # set D to reconstructed values and cast to double for mask indexing
                recon_t = tl.kruskal_to_tensor((normalize_scaling, [matrix for matrix in PhiY]))
                D = recon_t
                missing_mask, observed_mask = np.zeros(double_X.shape), np.ones(double_X.shape)
                missing_mask[(mask % (double_X.shape[0] * double_X.shape[1])) // double_X.shape[1],
                             (mask % (double_X.shape[0] * double_X.shape[1])) % double_X.shape[1],
                             mask // (double_X.shape[0] * double_X.shape[1])] = 1

                # d = (recon_t(:) + lambda * X(:)) * inv(1 + lambda)
                # D(setDifference) = d
                D = ((observed_mask - missing_mask) * (recon_t + lam[0] @ double_X)) / 1 + lam[0]
            else:  # watch out for this
                D = X
                recon_t = tl.kruskal_to_tensor((normalize_scaling, [matrix for matrix in PhiY]))
                # D([mask_i', mask_j', mask_t']) = mask(ktensor(normalize_scaling, PhiY), mask_tensor)
                # recover values from K tensor
                D[mask_i, mask_j, mask_t] = np.min(1, 1.01 ** i * 0.01) * \
                                            sparse.COO.from_numpy(tl.kruskal_to_tensor(
                                                (normalize_scaling, [matrix for matrix in PhiY]))) @ mask_tensor
        else:
            D = X

        time_one_iter = time.time()
        total_time_one_iter = time_one_iter - tic
        avg_time += total_time_one_iter
        if i % num_iters_check == 0:
            sparsity_constraint = 0
            for n in range(len(dimorder)):
                sparsity_constraint = sparsity_constraint + lam[n] * np.sum(np.abs(Y[n]))

            if not mask or mask_complex == 1:
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

    return None

# Function based doesn't work
def manual_run(file, X, psi_d, phi_d, mask):
    mat = load_matrix(file)
    
    match psi_d:
        case "RAM":
            psi_d = gen_rama(400, 10)
        case "GFT":
            psi_d = gen_gft(mat, False)[0]  # eigenvectors
        case "DFT":
            psi_d = gen_dft(200)
        case _:
            return
    
    match phi_d:
        case "RAM":
            phi_d = gen_rama(400, 10)
        case "GFT":
            phi_d = gen_gft(mat, False)[0]  # eigenvectors
        case "DFT":
            phi_d = gen_dft(200)
        case _:
            return

    # How do you use input x
    Y, W = tgsd(mat['X'], psi_d, phi_d, mask, iterations=100, K=7, lambda_1=.1, lambda_2=.1, lambda_3=1, rho_1=.01, rho_2=.01, type="rand")
    pred_matrix = psi_d @ Y @ W @ phi_d
    return pred_matrix

# Automatic
def config_run(config_path="config.json"):
    # Try to open the config file
    try:
        with open(config_path) as file:
            config = json.load(file)
    except FileNotFoundError:
        raise Exception(f"Config file '{config_path}' not found")
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON format in '{config_path}'")
    except Exception as e:
        raise Exception(f"Error loading config file: {e}")

    # Validate the mandatory keys
    if not ("psi" in config):
        raise Exception("Config must contain the 'psi' key")
    if not ("phi" in config):
        raise Exception("Config must contain the 'phi' key")
    if not ("x" in config):
        raise Exception("Config must contain the 'x' key")
    if not ("mask_mode" in config):
        raise Exception("Config must contain the 'mask_mode' key")
    if not ("mask_percent" in config):
        raise Exception("Config must contain the 'mask_percent' key")

    psi = str(config["psi"]).lower()
    phi = str(config["phi"]).lower()

    # Validate the runnability of the instance 
    if psi != "gft" and phi != "gft":
        raise Exception("At least one of PSI or PHI must be 'gft'")
    
    save_flag, load_flag = False, False
    
    # Check if the save flag in the config is enabled and validate the input 
    if "save_flag" in config:
        if not isinstance(config["save_flag"], bool):
            raise Exception("Invalid 'save_flag', must be a boolean")
        else:
            save_flag = config["save_flag"]

    # Check if the load flag in the config is enabled and validate the input
    if "load_flag" in config:
        if not isinstance(config["load_flag"], bool):
            raise Exception("Invalid 'load_flag', must be a boolean")
        else:
            load_flag = config["load_flag"]
    
    # Try to load the data
    try:
        data = np.genfromtxt(config["x"], delimiter=',', skip_header=1)
    except:
        raise Exception(f"Error loading data from '{config['x']}': {e}")
    
    match str(config["psi"]).lower():
        case "ram":
            #phi_d = gen_rama(400, 10)
            pass
        case "gft":
            #phi_d = gen_gft(mat, False)[0]  # eigenvectors
            pass
        case "dft":
            pass
            #phi_d = gen_dft(200)
        case _:
            raise Exception(f"PSI's dictionary, {config['psi']}, is invalid") 
        
    match str(config["phi"]).lower():
        case "ram":
            #phi_d = gen_rama(400, 10)
            pass
        case "gft":
            #phi_d = gen_gft(mat, False)[0]  # eigenvectors
            pass
        case "dft":
            pass
            #phi_d = gen_dft(200)
        case _:
            raise Exception(f"PHI's dictionary, {config['phi']}, is invalid") 
    
    # Validate the mask percent
    mask_percent = config["mask_percent"]
    if not (isinstance(mask_percent, int) or (mask_percent < 0 or mask_percent > 100)):
        raise Exception(f"{mask_percent} is invalid. Please enter a valid percent")
    
    # If the load flag is enabled load from file
    if(load_flag):
        # Retrieve the the correct path
        load_path = config["load_path"] if "load_path" in config else "save.match"
        # Try to load the data
        try:
            mask_data = np.loadtxt(load_path, dtype=float)
        except FileNotFoundError:
            raise Exception(f"Load path '{load_path}' does not exist")
    # If the load flag is not enabled check the mask mode
    else:
        # Validate and read the mask mode
        match str(config["mask_mode"]).lower():
            case "lin":
                mask_data = np.linspace(1, round(mask_percent/100 * data.size), round(mask_percent/100 * data.size))
            case "rand":
                mask_data = np.array(random.sample(range(1, data.size), round(mask_percent/100 * data.size)))
            case _:
                raise Exception(f"Invalid 'mask_mode': {config['mask_mode']}")
    
    # If the save flag is enabled save to file
    if(save_flag):
        # Retrieve the the correct path 
        save_path = config["save_path"] if "save_path" in config else "save.match"
        # Insure that data is not overwritten without user consent
        if(os.path.exists(save_path)):
            # If user permission is given try to write the data
            if("override" in config and config["override"]):
                try:
                    np.savetxt(save_path, mask_data)
                except Exception as e:
                    raise Exception(f"Error saving data: {e}")   
            # If user permission is not granted raise an exception 
            else:
                raise Exception(f"{save_path} already exists. Enable override to override the saved data.")
        # If the path does not already exist try to write the data
        else:
            try:
                np.savetxt(save_path, mask_data)
            except Exception as e:
                raise Exception(f"Error saving data: {e}")  
