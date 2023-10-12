import numpy as np
import scipy.io
import scipy.sparse.linalg as sla
from scipy.fftpack import fft
from scipy.sparse import diags
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
    Constructs a PsiGFT from matlab dictionary
    Args:
        dict (dict): Given matlab dictionary
        is_normalized (bool): Whether the matrix should be normalized
    Returns:
        list[np.ndarray]: list of numpy arrays in form [eigenvalues, eigenvectors]
    """
    adj = dict['adj'] # given adj matrix
    degree = np.array(adj.sum(axis=1)).flatten()

    # diagonal matrix
    D = diags(degree, format='csc')
    L = D - adj # Laplacian matrix

    # Matlab: [V, D] = eigs(A, B, sigma))
    # Python: [D, V] = eigs(A, sigma, B)  
    eigenvalues, psi_gft = sla.eigs(L) # to-do: eigenvalues only prints out the first 6
    if is_normalized: # normalize eigenvectors
        psi_gft /= psi_gft / np.linalg.norm(psi_gft, axis=0) # along the rows
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
    
# test against matlab code

mat = load_matrix() # load data
type = 'rand'
X_masked = 'X'
Psi_GFT = gen_gft(mat, False)
Psi_DFT = gen_dft(200)
print(Psi_DFT)