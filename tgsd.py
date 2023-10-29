import numpy as np
import scipy.io
from scipy.fftpack import fft
import scipy.sparse as sp
from math import gcd, pi
import cmath
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
    return scipy.io.loadmat('pyspady/demo_data.mat')
def gen_gft(dict: dict, is_normalized: bool) -> list[np.ndarray]:
    """
    Constructs a PsiGFT from matlab dictionary (for now)
    Args:
        dict (dict): Given matlab dictionary
        is_normalized (bool): Whether the matrix should be normalized
    Returns:
        list[np.ndarray]: list of numpy arrays in form [psi_gft, eigenvalues]
    """
    adj = dict['adj'] # given adj matrix
    #print(np.linalg.matrix_rank(adj.toarray()))
    # calculate sum along columns
    D = sp.diags(np.array(adj.sum(axis=0)).flatten())
    #print(np.linalg.matrix_rank(D.toarray()))
    L = D - adj # Laplacian matrix
    # normalize eigenvectors = D^-1/2*L*D^-1/2
    if is_normalized: 
        D_sqrt_inv = sp.diags(1.0 / np.sqrt(np.array(D.sum(axis=0)).flatten()))
        new_L = D_sqrt_inv @ L @ D_sqrt_inv
        eigenvalues, psi_gft = np.linalg.eig(new_L.toarray())
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        psi_gft = psi_gft[:, idx]
        return [psi_gft, eigenvalues]
        
    #print(np.linalg.matrix_rank(L.toarray()))
    eigenvalues, psi_gft = np.linalg.eig(L.toarray())
    # sort eigenvalues by ascending order such that constant vector is in first cell
    idx = np.argsort(eigenvalues)
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
    return (1/np.sqrt(t)) * fft(np.eye(t))

def gen_rama(t: int, max_period: int):
    """
    Constructs a Ramanujan periodic dictionary
    Args:
        t (int): Number of timesteps
        max_period (int): Number of max periods
    Returns:
        Ramanujan periodic dictionary of t timesteps (rows) with max_period
    """""
    A = np.array([])

    for n in range(1, max_period+1): # 1:max_period
        c1 = np.zeros((n, 1), dtype=complex)
        k_orig = np.arange(1, n+1)        
        k = [k__ for k__ in k_orig if gcd(k__, n) == 1]
        for n_ in range(n): # goes up to n-1 inclusive
            for a in k:
                c1[n_] = c1[n_] + cmath.exp(1j*2*pi*a*(n_)/n)
        c1 = np.real(c1)

        k_orig = np.arange(1, n+1)
        k = [k__ for k__ in k_orig if gcd(k__, n) == 1]
        CN_col_size = len(k) # size(k, 2) -> number of columns
        
        shifted_columns = []

        for j in range(1, CN_col_size + 1):
            shift = np.roll(c1, j - 1)
            shifted_columns.append(shift)
        # concatenate along the vertical axis
        CN = np.concatenate(shifted_columns, axis=1)

        #CNA = repmat(CN,floor(rowSize/N),1) 
        num_repeat = t//n
        CNA = np.tile(CN, (num_repeat, 1))
        #CN_cutoff = CN(1:rem(rowSize,N),:);
        remainder = t%n
        CN_cutoff = CN[:remainder, :]
        #CNA = cat(1,CNA,CN_cutoff);
        CNA = np.concatenate((CNA, CN_cutoff), axis=0)
        #A=cat(2,A,CNA);
        if A.size == 0: A = CNA
        else: A = np.concatenate((A, CNA), axis=1)
    return A

def tgsd(X: np.ndarray, psi_d: np.ndarray, psi_orth: bool, phi_d: np.ndarray, phi_orth: bool, mask: np.ndarray, termination_cond, 
         fit_tolerance, iterations: int, k: int, lambda_1: int, lambda_2: int, lambda_3: int, rho_1: int, rho_2: int):
    # both are orth; masked
    def update_d(p_P, p_X, p_mask, p_lambda_3):
        D = p_P.copy()

        # vectorize, remove missing values in x
        x = p_X.flatten()
        x[np.isin(x, p_mask)] = np.nan
        new_x = x[~np.isnan(x)].reshape(-1, 1)

        print(new_x.shape)

        # vectorize, remove missing values in p
        p = p_P.flatten()
        p[np.isin(p, p_mask)] = np.nan
        new_p = p[~np.isnan(p)].reshape(-1, 1)

        print(new_p.shape)

        # (p + lambda_3*x) * inv(1 + lambda_3)
        d = (new_p + np.dot(p_lambda_3, new_x)) / (1 + p_lambda_3)
        # D(setdiff(1:end, mask)) = d

        # TO-DO
        set_diff = np.setdiff1d(np.arange(len(D)), np.where(p_mask))
        D[set_diff] = d
        return D
    
    if mask.any():
        if psi_orth and phi_orth:
            n, t = X.shape
            hold, Y1 = psi_d.shape
            p, t = phi_d.shape
            I_2 = np.eye(n, p)
            W = np.zeros((k, p))
            W[0, 0] = 0.000001
            Y = np.zeros((Y1, k))
            sigma = np.eye(k, k)
            V, Z = 0, Y
            gamma_1, gamma_2 = Y, W
            obj_old, obj = 0, []
            I_Y = np.eye(np.dot(W.dot(phi_d), (W.dot(phi_d)).T).shape[0])
            I_W = np.eye(np.dot(psi_d.dot(Y).T, psi_d.dot(Y)).shape[0])
            for i in range(iterations):
                P = np.dot(np.dot(psi_d, Y), np.dot(W, phi_d))
                D = update_d(P, X, mask, lambda_3)
    return None

mat = load_matrix() # load data
ram = gen_rama(5, 10)
Psi_GFT = gen_gft(mat, False)
Psi_GFT = Psi_GFT[0] # eigenvectors
Psi_DFT = gen_dft(200)
tgsd(mat['X'], Psi_GFT, True, Psi_DFT, True, mat['mask'], None, None, 500, 7, .1, .1, .1, .01, .01)