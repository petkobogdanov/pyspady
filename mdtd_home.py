import numpy as np
import tensorly as tl
import scipy.sparse as sp
import time
import json
import mdtd_data_process
import dictionary_generation


class MDTD_Home:
    def __init__(self, config_path):
        self.X, self.adj_1, self.adj_2, self.mask, self.count_nnz, self.num_iters_check, self.lam, self.K, self.epsilon = self.mdtd_load_config(
            f"{config_path}")
        self.PhiY, self.recon_t = None, None

    def mdtd(self, is_syn: bool, X, adj1, adj2, mask, count_nnz=10, num_iters_check=0, lam=0.000001, K=10,
             epsilon=1e-4):
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
            syn_lambda = [0.000001 for _ in range(p_dim)]
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
            MDTD_Demo_Phi, MDTD_Demo_PhiYg, MDTD_Demo_P = dictionary_generation.GenerateDictionary.save_to_numpy_arrays()
            self.X = gen_syn_X(MDTD_Demo_PhiYg)  # numpy array of x * y * z
            num_modes = self.X.ndim
            lam, rho = gen_syn_lambda_rho(num_modes)  # list of size n
            phi_d = MDTD_Demo_Phi  # list of numpy arrays in form (1, n) where each atom corresponds to a dictionary.
            phi_type = ['not_ortho_dic', 'no_dic', 'no_dic']
            # first coordinate of each dictionary = shape of X
            P = MDTD_Demo_P  # Y values of shape of X
        else:
            X = self.X
            num_modes = X.ndim
            temp = lam
            lam = [temp] * num_modes if temp is not None else 0.000001
            rho = [val * 5 for val in lam]
            phi_d = np.empty((1, num_modes), dtype=object)

            for mode in range(num_modes):
                if mode == 0 or mode == 1:
                    nested_dictionary = dictionary_generation.GenerateDictionary.gen_gft_new(adj1, False)[
                        0] if mode == 0 else \
                        dictionary_generation.GenerateDictionary.gen_gft_new(adj2, False)[0]
                else:
                    nested_dictionary = dictionary_generation.GenerateDictionary.gen_rama(t=X.shape[2], max_period=24)
                phi_d[0, mode] = nested_dictionary

            # phi_type = [random.choice(['not_ortho_dic', 'ortho_dic', 'no_dic']) for _ in range(num_modes)]
            phi_type = ['no_dic', 'ortho_dic', 'no_dic']
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
            PhiYInit[n] = (phi_d[0, n] @ Y_init[n]) if phi_type[n] in ["not_ortho_dic", "ortho_dic"] else Y_init[n]
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
        for i in range(1, MAX_ITERS + 1):
            tic = time.time()  # start time
            for n in range(len(dimorder)):
                if phi_type[n] == "not_ortho_dic":
                    # calculate Unew = Phi X_(n) * KhatriRao(all U except n, 'r')
                    product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)
                    pv, Ev = np.linalg.eigh(product_vector)
                    pv = pv.reshape(-1, 1)

                    CC = PhiPhiEV[n].T @ (phi_d[0, n].T @ mttkrp(D, self.PhiY, n) + rho[n] * Z[n] - gamma[n]) @ Ev
                    Y[n] = CC / (rho[n] + PhiPhiLAM[n] @ pv.T)
                    Y[n] = PhiPhiEV[n] @ Y[n] @ Ev.T
                    self.PhiY[n] = phi_d[0, n] @ Y[n]
                    # normalize_scaling = sqrt(sum(Y{n}.^2, 1))' else max(max(abs(Y{n}), [], 1), 1)'
                    normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1, 1).T if i == 1 else np.maximum(
                        np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                    Y[n] /= normalize_scaling
                    self.PhiY[n] = phi_d[0, n] @ Y[n]
                    YPhi_Inner[:, :, n] = self.PhiY[n].T @ self.PhiY[n]

                elif phi_type[n] == "ortho_dic":
                    # phi_d_rao_other_factors = (phi_d[0, n].T @ mttkrp(D, PhiY, n) + rho[n] * Z[n] - gamma[n])
                    product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)
                    # denominator = product_vector + rho[n] * np.eye(K)
                    Y[n] = np.linalg.solve((product_vector + rho[n] * np.eye(K)).T,
                                           (phi_d[0, n].T @ mttkrp(D, self.PhiY, n) + rho[n] * Z[n] - gamma[n]).T).T
                    normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1, 1).T if i == 1 else np.maximum(
                        np.max(np.abs(Y[n]), axis=0), 1).reshape(-1, 1).T
                    Y[n] /= normalize_scaling
                    self.PhiY[n] = phi_d[0, n] @ Y[n]
                    YPhi_Inner[:, :, n] = Y[n].T @ Y[n]

                elif phi_type[n] == "no_dic":
                    Y[n] = mttkrp(D, self.PhiY, n)
                    # inversion_product_vector = np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)
                    Y[n] = np.linalg.solve(
                        (np.prod(YPhi_Inner[:, :, list(range(n)) + list(range(n + 1, num_modes))], axis=2)).T, Y[n].T).T
                    normalize_scaling = np.sqrt(np.sum(Y[n] ** 2, axis=0)).reshape(-1, 1).T if i == 1 else np.maximum(
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
                D = self.recon_t

                missing_mask, observed_mask = np.zeros(double_X.shape), np.ones(double_X.shape)

                # for idx in mask:
                #    nd_idx = nd_index(idx, double_X.shape)  # Convert index to correct tuple
                #    missing_mask[nd_idx] = 1
                nd_indices = np.unravel_index(mask, double_X.shape)
                missing_mask[nd_indices] = 1
                D = ((observed_mask - missing_mask) * (self.recon_t + lam[0] * double_X)) / 1 + lam[0]
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
                objs[i // num_iters_check, 1] = objs[i // num_iters_check - 1, 1] + total_time_one_iter  # 2, 2 and 1, 2
                fit_change = np.abs(objs[i // num_iters_check, 0] - objs[i // num_iters_check - 1, 0])

                print(
                    f"Iteration={i} Fit={fit} f-delta={fit_change} Reconstruction Error={recon_error} Sparsity Constraint={sparsity_constraint} Total={objs[i // num_iters_check, :]}")

                if fit_change < epsilon:
                    print(f"{avg_time}, {i}")
                    print(f"Algo has met fit change tolerance, avg time: {avg_time / i}")
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
            if config['adj-1']:
                adj_1 = build_adj_matrix(config["adj-1"], X.shape[0])
            if config['adj-2']:
                adj_2 = build_adj_matrix(config["adj-2"], X.shape[1])

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

        return X, adj_1, adj_2, mask, count_nnz, num_iters_check, lam, K, epsilon


if __name__ == '__main__':
    # mdtm_input_x, mdtm_input_adj, mdtm_input_mask, mdtm_input_count_nnz, mdtm_input_num_iters_check, mdtm_input_lam, mdtm_input_K, mdtm_input_epsilon = mdtm_load_config()
    mdtm_X, recon_X, phi_y = mdtd(is_syn=True, X=None, adj1=None, adj2=None, mask=[], count_nnz=0, num_iters_check=10,
                                  lam=0.000001, K=10,
                                  epsilon=1e-4)
    mdtd_clustering(phi_y, 5)
    # mdtm_find_outlier(mdtm_X, recon_X, 10)
