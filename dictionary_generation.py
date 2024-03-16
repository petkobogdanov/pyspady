import cmath
from math import gcd, pi
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.fftpack import fft
import pandas as pd
from scipy.interpolate import BSpline

class GenerateDictionary:
    @staticmethod
    def load_tgsd_demo() -> dict:
        """
        Loads matrix from file in directory
        Returns:
            dict: dictionary with variable names as keys, and loaded matrices as values
        """
        return scipy.io.loadmat('demo_data.mat')

    @staticmethod
    def load_mdtd_syn_data() -> dict:
        """
        Loads synthetic matrix from file in directory
        Returns:
            dict: dictionary with variable names as keys, and loaded matrices as values
        """

        def save_array(a, file):
            for i, a in enumerate(a):
                pd.DataFrame(a).to_csv(f"{file}_{i}.csv", index=False, header=False)

        def save_lists(l, file):
            for i, lst in enumerate(l):
                pd.DataFrame(lst).to_csv(f"{file}_{i}.csv", index=False, header=False)

        data = scipy.io.loadmat("syn_data.mat")
        # Phi is of size (1, x). For Phi_x, each index i.e. d[Phi[0, x]] corresponds to a separate numpy array to load into a .csv.
        # The numpy array will look like [phi_1, phi_2, phi_3] where phi_x represents the .csv file.
        # P is of size (1, x).Each index i.e. d[P[0, x]] corresponds to a list of a list, i.e. [[200]],[[300]],[[32]]
        # PhiYg follows the same format as Phi.
        if 'Phi' in data:
            phi_arrays = [data['Phi'][0, i] for i in range(data['Phi'].shape[1])]
            save_array(phi_arrays, "Phi")
        if 'P' in data:
            p_lists = [data['P'][0, i] for i in range(data['P'].shape[1])]
            save_lists(p_lists, "P")
        if 'PhiYg' in data:
            phiyg_arrays = [data['PhiYg'][0, i] for i in range(data['PhiYg'].shape[1])]
            save_array(phiyg_arrays, "PhiYg")

    @staticmethod
    def gen_gft_new(matrix, is_normalized: bool) -> list[np.ndarray]:
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

    @staticmethod
    def gen_dft(t: int) -> np.ndarray:
        """
        Constructs a PsiDFT
        Args:
            t (int): Number of timesteps
        Returns:
            np.ndarray: new DFT matrix
        """
        return (1 / np.sqrt(t)) * fft(np.eye(t))

    @staticmethod
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
        return A.T

    @staticmethod
    def gen_spline(p_signal_length, p_num_space=20, p_degree=4):
        def bspline_basis(i, p_degree, p_knot_vector, p_x):
            def bspline_basis_recurrence(i, p_degree, p_knot_vector, p_x):
                y = np.zeros(p_x.shape)

                if p_degree > 1:
                    b = bspline_basis_recurrence(i, p_degree - 1, p_knot_vector, p_x)
                    d_n = p_x - p_knot_vector[0, i]
                    d_d = p_knot_vector[0, i + p_degree - 1] - p_knot_vector[0, i]
                    if d_d != 0:
                        y += b * (d_n / d_d)

                    b = bspline_basis_recurrence(i + 1, p_degree - 1, p_knot_vector, p_x)
                    d_n = p_knot_vector[0, i + p_degree] - p_x
                    d_d = p_knot_vector[0, i + p_degree] - p_knot_vector[0, i + 1]

                    if d_d != 0:
                        y += b * (d_n / d_d)

                elif p_knot_vector[0, i+1] < p_knot_vector[0, -1]:
                    y[(p_knot_vector[0, i] <= p_x) & (p_x < p_knot_vector[0, i + 1])] = 1
                else:
                    y[p_knot_vector[0, i] <= p_x] = 1

                return y

            return bspline_basis_recurrence(i, p_degree, p_knot_vector, p_x)

        def bspline_basismatrix(p_degree, p_knot_vector, p_x):
            num_basis_functions = p_knot_vector.shape[1] - p_degree + 1
            B = np.zeros((p_x.shape[1], num_basis_functions))

            for i in range(num_basis_functions-1):
                B[:, i] = bspline_basis(i, p_degree, p_knot_vector, p_x)
            return B

        if isinstance(p_signal_length, int):
            s1, s2 = 1, p_signal_length
        else:
            s1, s2 = min(p_signal_length), max(p_signal_length)

        step = (s2-s1)/p_num_space
        knot_vector = np.arange(s1, s2 + step, step)
        knot_vector = np.concatenate(([s1] * (p_degree-1), knot_vector, [s2] * (p_degree-1)))
        knot_vector = knot_vector.reshape(1, len(knot_vector))
        x = np.arange(1, p_signal_length+1).reshape(1, p_signal_length)
        return bspline_basismatrix(p_degree, knot_vector, x)

