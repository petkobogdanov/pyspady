import numpy as np
import scipy.io
from scipy.fftpack import fft
import scipy.sparse as sp
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
    # calculate sum along columns
    D = sp.diags(np.array(adj.sum(axis=0)).flatten())
    L = D - adj # Laplacian matrix
    eigenvalues, psi_gft = np.linalg.eig(L.toarray()) 
    print(psi_gft[0])
    # normalize eigenvectors = D^1/2*L*D^1/2
    if is_normalized: psi_gft = np.dot(np.dot(np.sqrt(D), L), np.sqrt(D))
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