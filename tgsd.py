import scipy.io
from scipy.linalg import eigh
from scipy.fftpack import fft
import numpy as np

class TGSD:
    def __init__(self, iter, K, 
                lambda_1, lambda_2, lambda_3, rho_1, rho_2,
                mask, Psi_orth_1, Psi_orth_2) -> None:
        self.iter = iter                # 500
        self.K = K                      # 7
        self.lambda_1 = lambda_1        # .1
        self.lambda_2 = lambda_2        # .1
        self.lambda_3 = lambda_3        # 1
        self.rho_1 = self.lambda_1/10        # lambda_1/10
        self.rho_2 = self.lambda_2/10        # lambda_2/10
        self.mask = mask                # mask (vectorized location in 'rand' case)
        self.Psi_orth_1 = Psi_orth_1    # 1
        self.Psi_orth_2 = Psi_orth_2    # 1

    def load_matrix(self) -> dict:
        """
        Loads matrix from file in directory
        Returns:
            dict: dictionary with variable names as keys, and loaded matrices as values
        """
        return scipy.io.loadmat('pyspady/demo_data.mat')
    def build_psi_gft(self, dict: dict) -> list[np.ndarray]:
        """
        Constructs a PsiGFT from matlab dictionary
        Args:
            dict (dict): Given matlab dictionary
        Returns:
            list[np.ndarray]: list of numpy arrays in form [eigenvalues, eigenvectors]
        """
        adj = dict['adj'] # given adj matrix
        D = np.diag(np.sum(adj, axis=1)) # diagonal matrix
        L = D - adj # Laplacian matrix
        num_eigenvalues = 175
        eigenvalues, eigenvectors = eigh(L, subset_by_index=(0, num_eigenvalues-1))
        return [eigenvalues, eigenvectors]

    def build_psi_dft(self, size: int) -> np.ndarray:
        """
        Constructs a PsiDFT
        Args:
            size (int): Size of DFT matrix to construct
        Returns:
            np.ndarray: new DFT matrix
        """
        return (1/np.sqrt(size)) * fft(np.eye(size))
    

    """
    TODO
    %Sigma is unused
    [objs,Y,Sigma,W,V,Z]=optimization(X_masked,PsiGFT,PhiDFT,parm,type);

    %pred matrix with values reconstructed
    pred_matrix = PsiGFT*Y*W*PhiDFT;
    """

# test against matlab code

obj = TGSD(500, 7, 0.1, 0.1, 1, .1, .1, 'mask', 1, 1) # basic params
mat = obj.load_matrix() # load data
type = 'rand'
X_masked = 'X'
Psi_GFT = obj.build_psi_gft(mat)
Psi_DFT = obj.build_psi_dft(10)