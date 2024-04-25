import numpy as np
import scipy.io
import tensorly as tl
import scipy.sparse as sp
import time
import json
from mdtd_src import mdtd_data_process
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
import dictionary_generation

class MDTD_Home:
    def __init__(self, config_path):
        self.X, self.d1, self.d2, self.d3, self.adj_1, self.adj_2, self.mask, \
        self.count_nnz, self.num_iters_check, self.lam, self.K, self.epsilon = self.mdtd_load_config(f"{config_path}")
        self.PhiY, self.recon_t = None, None
        self.K = 20


    def mdtd(self, is_syn: bool, X, adj1, adj2, mask, count_nnz=10, num_iters_check=10, lam=0.000001, K=10,
             epsilon=1e-4, model="admm"):
        """
        Performs tensor decomposition across multi-dictionaries
        Args:
            is_syn: Boolean; whether MDTD will use synthetic test data
            X: User input, a tensor of n-mode
            adj1: Some user adjacency matrix as a numpy array
            adj2: Some user adjacency matrix as a numpy array
            mask: Mask of linear indexing to hide certain values of X from imputation/decomposition
            count_nnz: Number of non-zeros to count
            num_iters_check: Every number of iterations to calculate fit
            lam: Some user input lambda value as a list
            K: Some user input tensor rank of X
            epsilon: Some user input epsilon value
            model: Either Alternative Direction Method of Multipliers or Gradient Descent
            More information on these models is provided in the technical report.
        Returns:
            Original input tensor X, reconstructed tensor of X, Phi*Y
        """

        def gen_syn_X(p_PhiYg):
            """
            Generates synthetic X based on PhiYg from syn_data.mat
            Args:
                p_PhiYg: List of PhiYg (synthetic) from Matlab data
            Returns:
                Synthetic tensor X: Kruskal tensor of PhiYg
            """
            # Create a list of matrices from the cell array
            return tl.kruskal_to_tensor((np.ones(10), [matrix for matrix in p_PhiYg[0]]))

        def gen_syn_lambda_rho(p_dim):
            """
            Generates synthetic lambda and rho values
            Args:
                p_dim: Dimensions of the synthetic tensor X

            Returns:
                List of synthetic lambda values and list of synthetic rho values
            """
            syn_lambda = [0.0001 for _ in range(p_dim)]
            syn_rho = [val * 5 for val in syn_lambda]
            return syn_lambda, syn_rho

        def mttkrp(p_D, p_PhiY, p_n):
            """
            Computes Matricized Tensor Times Khatri-Rao Product
            Args:
                p_D: Reconstruction of tensor X
                p_PhiY: List of matrices of shape (X.shape[i], K)
                p_n: Mode along which to vectorize p_D

            Returns:
                The Matricized Tensor Times Khatri-Rao Product of p_D and p_PhiY along the n-th mode
            """
            return tl.unfold(p_D, mode=p_n).astype('float32') @ tl.tenalg.khatri_rao(
                [p_PhiY[i].astype('float32') for i in range(len(p_PhiY)) if i != p_n])

        if is_syn:
            # Generate synthetic data numpy arrays
            MDTD_Demo_Phi, MDTD_Demo_PhiYg, MDTD_Demo_P = mdtd_data_process.MDTD_Data_Process.syn_data_to_numpy()
            self.X = gen_syn_X(MDTD_Demo_PhiYg)  # numpy array of x * y * z

            X = self.X
            num_modes = self.X.ndim
            lam, rho = gen_syn_lambda_rho(num_modes)  # list of size n
            phi_d = MDTD_Demo_Phi

            # list of numpy arrays in form (1, n) where each atom corresponds to a dictionary.
            phi_type = ['ortho_dic', 'not_ortho_dic', 'not_ortho_dic']
            # first coordinate of each dictionary = shape of X
            P = MDTD_Demo_P  # Y values of shape of X

            num_to_mask = int(np.prod(X.shape) * (1 / 100.0))
            all_indices = np.arange(np.prod(X.shape))
            mask = np.random.choice(all_indices, num_to_mask, replace=False)
            self.mask = mask

        else:
            X = self.X
            num_modes = X.ndim
            temp = lam
            lam = [temp] * num_modes if temp is not None else 0.000001
            rho = [val * 5 for val in lam]
            phi_d = np.empty((1, num_modes), dtype=object)

            for mode in range(num_modes):
                if mode == 0:
                    nested_dictionary = self.d1
                elif mode == 1:
                    nested_dictionary = self.d2
                else:
                    nested_dictionary = self.d3
                phi_d[0, mode] = nested_dictionary

            # phi_type = [random.choice(['not_ortho_dic', 'ortho_dic', 'no_dic']) for _ in range(num_modes)]
            phi_type = ['ortho_dic', 'ortho_dic', 'no_dic']
            P = np.empty((1, num_modes), dtype=object)

            for i in range(phi_d.shape[1]):
                if isinstance(phi_d[0, i], list):
                    y_idx = phi_d[0, i][0].shape[1]
                else:
                    y_idx = phi_d[0, i].shape[1]
                dtype = np.uint8 if y_idx <= 250 else np.uint16
                P[0, i] = np.array([y_idx], dtype=dtype)

        # In the case of no dictionary, edit P manually
        for _ in range(len(phi_type)):
            num = X.shape[_]
            if phi_type[_] == "no_dic":
                P[0, _] = np.array([num], dtype=np.uint16)

        # cast to Double for mask indexing i.e. double_X = double(X)
        K = self.K

        print(f"Phi types in this run: {phi_type}")
        D = X
        normalize_scaling = np.ones(K)
        dimensions = X.shape
        mask_complex = 1
        if ((isinstance(mask, np.ndarray) and mask.any()) or (
                isinstance(mask, list) and len(mask) > 0)) and mask_complex == 1:
            double_X = X.astype(np.longdouble)

        np.random.seed(6)

        PhiPhiEV, PhiPhiLAM = [None] * num_modes, [None] * num_modes
        # dictionary decomposition
        for i in range(num_modes):
            if phi_type[i] == "not_ortho_dic":
                # U and V are unitary matrices and S contains singular values as a diagonal matrix
                U, S_non_diag, V_transpose = np.linalg.svd(phi_d[0, i])
                S = np.diag(S_non_diag)
                V = V_transpose.T
                PhiPhiEV[i] = V
                retained_rank = min(phi_d[0, i].shape)
                # diag(S(1:retained_rank, 1:retained_rank)).^2
                PhiPhiLAM[i] = (np.diag(S[:retained_rank]) ** 2).reshape(-1, 1)

        MAX_ITERS = 500
        # 1:num_modes, @(x) isequal(sort(x), 1:num_modes)
        dimorder = np.arange(1, num_modes + 1)
        gamma = [None] * num_modes
        YPhiInitInner = np.zeros((K, K, num_modes))
        Y_init, PhiYInit = [None] * num_modes, [None] * num_modes

        for n in range(len(dimorder)):
            Y_init[n] = np.random.rand(P[0, n][0], K)
            PhiYInit[n] = (phi_d[0, n] @ Y_init[n]) if phi_type[n] in ["not_ortho_dic", "ortho_dic"] else Y_init[
                n]
            YPhiInitInner[:, :, n] = PhiYInit[n].T @ PhiYInit[n] if phi_type[n] == "not_ortho_dic" else Y_init[n].T @ \
                                                                                                        Y_init[n]
            gamma[n] = 0

        # set up for initialization, U and the fit
        Y = Y_init
        self.PhiY = PhiYInit
        YPhi_Inner = YPhiInitInner
        Z = Y
        self.recon_t = tl.kruskal_to_tensor((normalize_scaling, [matrix for matrix in self.PhiY]))
        normX = tl.norm(tl.tensor(X), order=2)
        objs = np.zeros((MAX_ITERS, num_modes))
        objs[0, 0] = tl.sqrt(
            (normX ** 2) + (tl.norm(self.recon_t, order=2) ** 2) - 2 * tl.tenalg.inner(X, self.recon_t))
        # iterate until convergence
        avg_time = 0

        if model == "admm":
            for i in range(1, MAX_ITERS + 1):
                tic = time.time()  # start time
                for n in range(len(dimorder)):
                    if phi_type[n] == "not_ortho_dic":
                        # calculate Unew = Phi X_(n) * KhatriRao(all U except n, 'r')
                        product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))],
                                                 axis=2)
                        pv, Ev = np.linalg.eigh(product_vector)
                        pv = pv.reshape(-1, 1)

                        CC = PhiPhiEV[n].T @ (phi_d[0, n].T @ mttkrp(D, self.PhiY, n) + rho[n] * Z[n] - gamma[n]) @ Ev
                        Y[n] = CC / (rho[n] + PhiPhiLAM[n] @ pv.T)
                        Y[n] = PhiPhiEV[n] @ Y[n] @ Ev.T
                        self.PhiY[n] = phi_d[0, n] @ Y[n]
                        # normalize_scaling = sqrt(sum(Y{n}.^2, 1))' else max(max(abs(Y{n}), [], 1), 1)'
                        normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1,
                                                                                       1).T if i == 1 else np.maximum(
                            np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                        Y[n] /= normalize_scaling
                        self.PhiY[n] = phi_d[0, n] @ Y[n]
                        YPhi_Inner[:, :, n] = self.PhiY[n].T @ self.PhiY[n]

                    elif phi_type[n] == "ortho_dic":
                        # phi_d_rao_other_factors = (phi_d[0, n].T @ mttkrp(D, PhiY, n) + rho[n] * Z[n] - gamma[n])
                        product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))],
                                                 axis=2)
                        # denominator = product_vector + rho[n] * np.eye(K)
                        other_factors = (
                                    (phi_d[0, n].T @ mttkrp(D, self.PhiY, n)) + rho[n] * csr_matrix(Z[n]) - gamma[n]).T

                        # Y[n] = np.linalg.solve((product_vector + rho[n] * np.eye(K)).T,
                        #                       (phi_d[0, n].T @ mttkrp(D, self.PhiY, n) + rho[n] * Z[n] - gamma[n]).T).T
                        Y[n] = spsolve((csr_matrix(product_vector) + rho[n] * scipy.sparse.eye(K, format='csr').T),
                                       other_factors).T

                        normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1,
                                                                                       1).T if i == 1 else np.maximum(
                            np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                        Y[n] /= normalize_scaling
                        self.PhiY[n] = phi_d[0, n] @ Y[n]
                        YPhi_Inner[:, :, n] = Y[n].T @ Y[n]

                    elif phi_type[n] == "no_dic":
                        product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))],
                                                 axis=2)
                        Y[n] = mttkrp(D, self.PhiY, n)
                        # inversion_product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)
                        Y[n] = spsolve(csr_matrix(product_vector).T, csr_matrix(Y[n]).T).T
                        Y[n] = Y[n].toarray()
                        # Y[n] = np.linalg.solve(
                        #    (np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)).T, Y[n].T).T
                        normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1,
                                                                                       1).T if i == 1 else np.maximum(
                            np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                        Y[n] /= normalize_scaling
                        self.PhiY[n] = Y[n]
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

                if (isinstance(mask, list) and len(mask) > 0) or (isinstance(mask, np.ndarray) and mask.any()):
                    if isinstance(mask, list): mask = np.array(mask)
                    # set D to reconstructed values and cast to double for mask indexing
                    self.recon_t = tl.kruskal_to_tensor(
                        (normalize_scaling.reshape((normalize_scaling.shape[1],)), [matrix for matrix in self.PhiY]))
                    D = X.copy()  # Make a copy of the input tensor

                    # Imputation:
                    mask_i, mask_j, mask_t = np.unravel_index(mask, D.shape)
                    D[mask_i, mask_j, mask_t] = tl.tensor(self.recon_t)[mask_i, mask_j, mask_t]
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
                        self.recon_t = tl.kruskal_to_tensor(
                            (np.squeeze(normalize_scaling), [matrix for matrix in self.PhiY]))

                    recon_error = tl.sqrt(
                        (normX ** 2) + (tl.norm(self.recon_t, order=2) ** 2) - 2 * tl.tenalg.inner(X, self.recon_t))

                    fit = 1 - (recon_error / normX)

                    objs[i // num_iters_check, 0] = fit  # 2, 1 == 1, 0
                    objs[i // num_iters_check, 1] = objs[
                                                        i // num_iters_check - 1, 1] + total_time_one_iter  # 2, 2 and 1, 2
                    fit_change = np.abs(objs[i // num_iters_check, 0] - objs[i // num_iters_check - 1, 0])

                    print(
                        f"Iteration={i} Fit={fit} f-delta={fit_change} Reconstruction Error={recon_error} Sparsity Constraint={sparsity_constraint} Total={objs[i // num_iters_check, :]}")

                    if fit_change < epsilon:
                        print(f"Total time: {avg_time}, Iteration: {i}")
                        print(f"Algo has met fit change tolerance, avg time per 1 iteration: {avg_time / i}")
                        break
        else:
            # Gradient Descent
            """
            For each iteration i from 1 to MAX_ITERS:
            a. Start timing the iteration.
            b. For each mode n:
                i. Compute the gradient grad_Y[n] with respect to the factor matrix Y[n].
                ii. Update the factor matrix Y[n] using the gradient descent rule: Y[n] = Y[n] - alpha * grad_Y[n].
            c. If i is a multiple of num_iters_check:
                i. Compute the reconstructed tensor recon_t from Y and phi_d.
                ii. If a mask is provided, apply it to X before computing the loss.
                iii. Compute the loss function value based on the difference between X and recon_t.
                iv. Store the loss value in objs for convergence checking.
                v. Print the current iteration, the loss, and other relevant metrics.
                vi. Check if the change in loss is below the threshold epsilon. If so, break the loop.
            d. End timing the iteration and update avg_time.
            """
            def update_reconstruction(X, Y, mask=None):
                # Apply weights to tensor reconstruction.
                # This takes the max. absolute value along axis=0 of each factor matrix in Y.
                normalize_scaling = np.squeeze(np.max(np.abs(np.concatenate(Y, axis=0)), axis=0, keepdims=True))

                # Impute missing values.
                self.recon_t = mask * X + (1 - mask) * tl.kruskal_to_tensor((normalize_scaling, self.PhiY))

                return self.recon_t

            LEARNING_RATE = 1e-8
            tol = 1e-6
            coef_threshold = 1e-3
            D = X.copy()
            tensor_mask = (np.ones(D.shape)).ravel()
            tensor_mask[mask] = 0
            tensor_mask = tensor_mask.reshape(D.shape)

            for iter in range(1, MAX_ITERS + 1):
                # Update PhiY:
                for n in range(num_modes):
                    D = update_reconstruction(X, Y, tensor_mask)
                    # residual = D_i.T - phi_d[0, n] * Y_i * (A . B)
                    # D = approximation of X.
                    # D_i is an unfolding along mode i.
                    # A = ΦjYj , B = ΦlYl where j < l.
                    # A . B denotes the Khatri-Rao product of A and B
                    # gradient = lambda[n] * sign[Y_n] - phi_d[0, n].T * residual * (A . B).T
                    # Already have a method for MTTKRP.

                    # tl.unfold(D, n) = 200x120000
                    # PhiY[n] = 200x20
                    # khatri rao = 120000x20
                    residual = tl.unfold(D, n) - self.PhiY[n] @ (tl.tenalg.khatri_rao(self.PhiY, skip_matrix=n)).T  # 200x120000


                    # 200x200 . 200x120000 . 120000x20
                    grad_Y = lam[n] * np.sign(Y[n]) - (phi_d[0, n].T @ residual @ (tl.tenalg.khatri_rao(self.PhiY, skip_matrix=n)))

                    Y_next = Y[n] - LEARNING_RATE * grad_Y
                    # Pruning.
                    Y_next[np.abs(Y_next) < coef_threshold] = 0
                    Y[n] = Y_next
                    # Compute PhiY after updating Y.
                    if phi_type[n] in ['not_ortho_dic', 'ortho_dic']:
                        self.PhiY[n] = phi_d[0, n] @ Y[n]
                    else:
                        self.PhiY[n] = Y[n]

                # Reconstruct tensor using PhiY updates.
                D = update_reconstruction(X, Y, tensor_mask)

                if iter % 10 == 0:
                    recon_error = tl.sqrt((normX ** 2) + (tl.norm(self.recon_t, order=2) ** 2) - 2 * tl.tenalg.inner(X, self.recon_t))

                    fit = 1 - (recon_error / normX)

                    objs[iter // num_iters_check, 0] = fit  # 2, 1 == 1, 0
                    objs[iter // num_iters_check, 1] = objs[iter // num_iters_check - 1, 1]  # 2, 2 and 1, 2
                    fit_change = np.abs(objs[iter // num_iters_check, 0] - objs[iter // num_iters_check - 1, 0])

                    print(f"Iteration={iter} Fit={fit} f-delta={fit_change} Reconstruction Error={recon_error} Total={objs[i // num_iters_check, :]}")

                    if fit_change < tol:
                        print(f"Gradient Descent converged at iteration {iter}")
                        break

        # [[S ⊡ Φ1Y1, Φ2Y2, Φ3Y3]
        return self.X, self.recon_t, self.PhiY

    def mdtd_load_config(self, p_filename):
        """
        Loads the config file mdtd_config.json and extracts expected parameters/hyperparameters based on user input
        Returns:
            Parameters for MDTD (X, adj_1, adj_2, mask, count_nnz, num_iters_check, lam, K, epsilon)
        """

        def build_adj_matrix(p_path, p_dim):
            """
            Constructs a square adjacency matrix given some path to an adjacency matrix .csv of shape p_dim by p_dim
            Args:
                p_path: Path to some .csv file of an adjacency matrix
                p_dim: Dimension of square adjacency matrix

            Returns:
                Sparse adjacency matrix of p_dim by p_dim
            """
            adj_data = np.loadtxt(p_path, delimiter=',', dtype=int)
            rows, cols = adj_data[:, 0], adj_data[:, 1]
            return sp.csc_matrix((np.ones_like(rows - 1), (rows - 1, cols - 1)), shape=(p_dim, p_dim))

        with open(f"{p_filename}", 'r') as file:
            config = json.load(file)

        # Load X from a CSV file
        if config['X']:
            X = mdtd_data_process.MDTD_Data_Process.mdtd_format_csv_to_numpy(config['X'])
        if X.any():
            # Construct adjacency matrices
            if config['dictionary-1'] == "GFT" and config['adj-1']:
                adj_1 = build_adj_matrix(config["adj-1"], X.shape[0])
                d1 = dictionary_generation.GenerateDictionary.gen_gft_new(adj_1, False)[0]
            if config['dictionary-2'] == "GFT" and config['adj-2']:
                adj_2 = build_adj_matrix(config["adj-2"], X.shape[1])
                d2 = dictionary_generation.GenerateDictionary.gen_gft_new(adj_2, False)[0]
            elif config["dictionary-2"] == "rama":
                d2 = dictionary_generation.GenerateDictionary.gen_rama(t=X.shape[1], max_period=24)
            elif config["dictionary-2"] == "spline":
                d2 = dictionary_generation.GenerateDictionary.gen_spline(p_signal_length=X.shape[1])
            if config["dictionary-3"] == "rama":
                d3 = dictionary_generation.GenerateDictionary.gen_rama(t=X.shape[2], max_period=24)
            else:
                d3 = dictionary_generation.GenerateDictionary.gen_spline(p_signal_length=X.shape[2])

        # Construct a random, linear indexed mask based on X shape
        mask_percentage = config["mask_percentage_random"]
        if mask_percentage > 0:
            num_to_mask = int(np.prod(X.shape) * (mask_percentage / 100.0))
            all_indices = np.arange(np.prod(X.shape))
            mask = np.random.choice(all_indices, num_to_mask, replace=False)
        else:
            mask = np.array([])

        # The rest of the configuration parameters
        count_nnz = config['count_nnz']
        num_iters_check = config['num_iters_check']
        lam = config['lam']
        K = config['K']
        epsilon = config['epsilon']

        return X, d1, d2, d3, adj_1, adj_2, mask, count_nnz, num_iters_check, lam, K, epsilon

    @staticmethod
    def return_missing_values(p_mask, p_recon_t):
        """
        Reconstructs tensor and downloads .csv with missing values from mask.
        Args:
            p_mask: Given mask as linear indices.
            p_recon_t: Given reconstructed tensor.

        Returns:
            .csv file of missing values of the form {mode, row_index, col_index, value}.
        """
        row_indices, col_indices, depth_indices = np.unravel_index(p_mask, p_recon_t.shape)
        imputed_values = p_recon_t[row_indices, col_indices, depth_indices]
        depth_indices = depth_indices.flatten()
        row_indices = row_indices.flatten()
        col_indices = col_indices.flatten()
        imputed_values = imputed_values.flatten()
        df = pd.DataFrame({
            "Row #": row_indices,
            "Col #": col_indices,
            "Depth #": depth_indices,
            "Imputed Value": imputed_values
        })
        csv_path = "../tensor_imputed_values.csv"
        df.to_csv(csv_path, index=False)
        csv_path
