import math

import numpy as np
import scipy.io
from scipy.fftpack import fft
import scipy.sparse as sp
from math import gcd, pi
import cmath

from scipy.sparse.linalg import eigsh

"""""
        self.iter = iter                # 500
        self.K = K                      # 7
        self.lambda_1 = lambda_1        # .1
        self.lambda_2 = lambda_2        # .1
        self.lambda_3 = lambda_3        # 1
        self.rho_1 = self.lambda_1/10   # lambda_1/10
        self.rho_2 = self.lambda_2/10   # lambda_2/10
        self.mask = mask                # mask (vectorized location in 'rand' case)
        self.Psi_orth_1 = Psi_orth_1    # 1
        self.Psi_orth_2 = Psi_orth_2    # 1
"""""


def load_matrix() -> dict:
    """
    Loads matrix from file in directory
    Returns:
        dict: dictionary with variable names as keys, and loaded matrices as values
    """
    return scipy.io.loadmat('demo_data.mat')

def gen_gft(dict: dict, is_normalized: bool) -> list[np.ndarray]:
    """
    Constructs a PsiGFT from matlab dictionary (for now)
    Args:
        dict (dict): Given matlab dictionary
        is_normalized (bool): Whether the matrix should be normalized
    Returns:
        list[np.ndarray]: list of numpy arrays in form [psi_gft, eigenvalues]
    """
    adj = dict['adj']  # given adj matrix
    # print(np.linalg.matrix_rank(adj.toarray()))
    # calculate sum along columns
    D = sp.diags(np.array(adj.sum(axis=0)).flatten())
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
    eigenvalues, psi_gft = np.linalg.eigh(L.toarray())
    # sort eigenvalues by ascending order such that constant vector is in first cell
    idx = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    psi_gft = psi_gft[:, idx]

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
                c1[n_] = c1[n_] + cmath.exp(1j * 2 * pi * a * (n_) / n)
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


def tgsd(X: np.ndarray, psi_d: np.ndarray, phi_d: np.ndarray, mask: np.ndarray,
         termination_cond,
         fit_tolerance, iterations: int, k: int, lambda_1: int, lambda_2: int, lambda_3: int, rho_1: int, rho_2: int):
    def is_orthonormal(p_psi_or_phi):
        """
        Determines if graph or time series dictionary is orthonormal
        Args:
            p_psi_or_phi: Specified graph or time series dictionary
        Returns:
            True if orthonormal, false otherwise
        """
        return np.allclose(np.dot(p_psi_or_phi.T, p_psi_or_phi), np.eye(p_psi_or_phi.shape[1]), atol=1e-10)
    def update_d(p_P, p_X, p_mask, p_lambda_3):
        """
        Learns D via P, X, mask, and lambda 3
        Args:
            p_P: ΨYWΦ
            p_X: Input X
            p_mask: Indices represent masked indices in D
            p_lambda_3: Some lambda value

        Returns:
            D = (P +𝜆3Ω ⊙ X) ⊘ (I +𝜆3Ω)
        """
        p_mask, missing_mask, observed_mask = p_mask-1, np.zeros(p_P.shape), np.ones(p_P.shape)
        missing_mask[p_mask % p_P.shape[0], p_mask // p_P.shape[0]] = 1
        return ((p_P + p_lambda_3 * p_X) / (1+p_lambda_3) * (observed_mask-missing_mask)) + (p_P * missing_mask)

    def get_object(p_mask, p_D, p_X, p_phi, p_psi, p_Y, p_sigma, p_W, p_lambda_1, p_lambda_2, p_lambda_3):
        """
        Returns a new object represented by equation X−YΨWΦ‖
        Args:
            p_mask: Some specified mask
            p_D: Some specified D
            p_X: Some specified X
            p_phi: Some specified Phi dictionary
            p_psi: Some specified Psi dictionary
            p_Y: Some specified Y
            p_sigma: Some specified Sigma
            p_W: Some specified W
            p_lambda_1: Some specified lambda 1 value
            p_lambda_2: Some specified lambda 2 value
            p_lambda_3: Some specified lambda 2 value

        Returns:
            New object represented by X−YΨWΦ‖
        """
        # temp=X(:)-D(:);
        # term3=norm(temp(setdiff(1:end,mask)));

        p_mask, missing_mask, observed_mask = p_mask-1, np.zeros(p_X.shape), np.ones(p_X.shape)
        missing_mask[p_mask % 175, p_mask // 175] = 1
        term_3 = np.linalg.norm((observed_mask-missing_mask) * (p_D-p_X))
        #obj_new = norm(D-Psi*Y*Sigma*W*Phi)+lambda_1*norm(Y,1)+lambda_2*norm(W,1)+lambda_3*term3;
        return np.linalg.norm(p_D - p_psi @ p_Y @ p_sigma @ p_W @ p_phi) \
                  + p_lambda_1 * np.linalg.norm(p_Y, ord=1) + p_lambda_2 * np.linalg.norm(p_W, ord=1) + p_lambda_3 * term_3
    if mask.any():
        # both are orth; masked
        if is_orthonormal(psi_d) or is_orthonormal(phi_d):
            n, t = X.shape
            hold, Y1 = psi_d.shape
            p, t = phi_d.shape
            I_2 = np.eye(n, p)
            W = np.zeros((k, p))
            W[0, 0] = .000001
            Y = np.zeros((Y1, k))
            sigma = np.eye(k, k)
            V, Z = 0, Y
            gamma_1, gamma_2 = Y, W
            obj_old, objs = 0, []
            I_Y = np.eye((W @ phi_d @ (W @ phi_d).T).shape[0])
            I_W = np.eye(((psi_d @ Y).T @ psi_d @ Y).shape[0])

            for i in range(1, 1+iterations):
                P = psi_d @ Y @ W @ phi_d
                D = update_d(P, X, mask, lambda_3)
                #     B=Sigma*W*Phi;
                #     Y=(2*Psi'*D*B'+rho_1*Z+Gamma_1)*inv(2*(B*B')+rho_1*I_y+exp(-15));
                B = sigma @ W @ phi_d
                Y = (2 * psi_d.T @ D @ B.T + rho_1 * Z + gamma_1) @ np.linalg.inv(2 * (B @ B.T) + rho_1 * I_Y + math.exp(-15))
                # Update Z:
                # h = Y-gamma_1 / rho_1
                # Z = sign(h).*max(abs(h)-lambda_1/rho_1, 0)
                h = Y - gamma_1 / rho_1
                Z = np.sign(h) * np.maximum(np.abs(h) - lambda_1 / rho_1, 0)
                # A = psi*Y*sigma
                A = psi_d @ Y
                # W = inv(2*(A')*A + I_W*rho_2) * (2*A'*D*phi'+rho_2*V+gamma_2)
                W_first = np.linalg.inv(2 * A.T @ A + I_W * rho_2)
                W_final = 2 * A.T @ D @ phi_d.T + rho_2 * V + gamma_2
                W = W_first @ W_final
                # Update V:
                # h= W-Gamma_2/rho_2;
                # V = sign(h).*max(abs(h)-lambda_2/rho_2,0);
                h = W - gamma_2 / rho_2
                V = np.sign(h) * np.maximum(np.abs(h)-lambda_2/rho_2, 0)

                gamma_1, gamma_2 = gamma_1 + rho_1*(Z-Y), gamma_2 + rho_2*(V-W)
                rho_1, rho_2 = min(rho_1*1.1, 1e5), min(rho_2*1.1, 1e5)
                # Stop condition
                if i % 25 == 0:
                    obj_new = get_object(mask, D, X, phi_d, psi_d, Y, sigma, W, lambda_1, lambda_2, lambda_3)
                    objs = [objs, obj_new]
                    residual = abs(obj_old-obj_new)
                    print(f"obj-{i}={obj_new}, residual-{i}={residual}")
                    if residual < 1e-6: break
                    else: obj_old = obj_new
    return None


mat = load_matrix()  # load data
ram = gen_rama(5, 10)
Psi_GFT = gen_gft(mat, False)
Psi_GFT = Psi_GFT[0]  # eigenvectors
Psi_DFT = gen_dft(200)
tgsd(mat['X'], Psi_GFT, Psi_DFT, mat['mask'], None, None, 2, 7, .1, .1, 1, .01, .01)
