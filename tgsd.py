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
    #print(np.linalg.matrix_rank(L.toarray()))
    eigenvalues, psi_gft = np.linalg.eig(L.toarray())
    # sort eigenvalues by ascending order such that constant vector is in first cell
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    psi_gft = psi_gft[:, idx] 
    print(eigenvalues)
    print(psi_gft)
    # normalize eigenvectors = D^1/2*L*D^1/2
    if is_normalized: psi_gft = np.dot(np.dot(np.sqrt(D), L.toarray()), np.sqrt(D))
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
    """

    """""
    CNA = repmat(CN,floor(rowSize/N),1);
    CN_cutoff = CN(1:rem(rowSize,N),:);
    CNA =cat(1,CNA,CN_cutoff);
    """""
    A = np.empty((t, 0))

    for n in range(1, max_period+1): # 1:max_period
        c1 = np.zeros((n, 1), dtype=complex)
        k_orig = np.arange(1, n+1)
        k = [k__ for k__ in k_orig if gcd(k__, n) == 1]
        for n_ in range(n): # goes up to n-1
            for a in k:
                c1[n_] = c1[n_] + cmath.exp(1j*2*pi*a*(n_)/n)
        c1 = np.real(c1)

        k_orig = np.arange(1, n+1)
        k = [k__ for k__ in k_orig if gcd(k__, n) == 1]
        CN_col_size = len(k) # size(k, 2) -> number of columns
        shifted_arrays = []
        for j in range(1, CN_col_size+1): # 1:CN_col_size
            c1_circshift = np.roll(c1, j-1, axis=0)
            shifted_arrays.append(c1_circshift) # CN = cat(2,CN,circshift(c1,(j-1)))

        CN = np.hstack(shifted_arrays) # stack by column
        CNA = np.full((t // n, 1), CN) # repmat(CN,floor(rowSize/N),1)
        rem = t % n # rem(rowSize,N)
        CN_cutoff = CN[1:rem+1, :] # CN(1:rem(rowSize,N),:)

        CNA = np.concatenate((CNA, CN_cutoff), axis=0) # CNA = cat(1,CNA,CN_cutoff)

        A = np.hstack((A, CNA)) # A = cat(2,A,CNA)
# test against matlab code

mat = load_matrix() # load data
type = 'rand'
X_masked = 'X'
#ram = gen_rama(5, 10)
Psi_GFT = gen_gft(mat, False)
Psi_DFT = gen_dft(200)