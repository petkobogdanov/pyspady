import numpy as np
import scipy.io
import tensorly as tl
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
import json
import pandas as pd
from tgsd import gen_gft_new, gen_rama
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def load_syn_data() -> dict:
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


def save_to_numpy_arrays():
    load_phi_array = []
    for i in range(3):  # Assuming you have 3 files to load
        arr = np.loadtxt(f'mdtd_demo_data/Phi_{i}.csv', delimiter=',')
        load_phi_array.append(arr)
    phi_array = np.empty((1, 3), dtype='object')
    for i, arr in enumerate(load_phi_array):
        phi_array[0, i] = arr

    load_phi_yg_array = []
    for i in range(3):  # Assuming you have 3 files to load
        arr = np.loadtxt(f'mdtd_demo_data/PhiYg_{i}.csv', delimiter=',')
        load_phi_yg_array.append(arr)
    phi_yg_array = np.empty((1, 3), dtype='object')
    for i, arr in enumerate(load_phi_yg_array):
        phi_yg_array[0, i] = arr

    list_string = pd.read_csv(f'mdtd_demo_data/P.csv', header=None).iloc[0]
    p_array_of_arrays = np.empty((list_string.size,), dtype=object)
    for i, item in enumerate(list_string):
        p_array_of_arrays[i] = np.array([item], dtype=np.uint16)

    # Reshape to 1xN where each element is a numpy array
    p_array_of_arrays = p_array_of_arrays.reshape(1, -1)

    return phi_array, phi_yg_array, p_array_of_arrays


def mdtd(is_syn: bool, X, adj1, adj2, mask, count_nnz=10, num_iters_check=0, lam=0.000001, K=10, epsilon=1e-4):
    def gen_syn_X(p_PhiYg):
        # Create a list of matrices from the cell array
        return tl.kruskal_to_tensor((np.ones(10), [matrix for matrix in p_PhiYg[0]]))

    def gen_syn_lambda_rho(p_dim):
        syn_lambda = [0.000001 for _ in range(p_dim)]
        syn_rho = [val * 5 for val in syn_lambda]
        return syn_lambda, syn_rho

    def mttkrp(p_D, p_PhiY, p_n):
        # Matricized tensor times Khatri-Rao product
        return tl.unfold(p_D, mode=p_n).astype('float32') @ tl.tenalg.khatri_rao(
            [p_PhiY[i].astype('float32') for i in range(len(p_PhiY)) if i != p_n])

    def nd_index(index, shape):
        indices = [0] * len(shape)
        for i in range(len(shape) - 1, -1, -1):
            stride = np.prod(shape[:i])
            indices[i], index = divmod(index, stride)
            indices[i] = int(indices[i])
        return tuple(indices)

    if is_syn:
        # Generate synthetic data numpy arrays
        MDTD_Demo_Phi, MDTD_Demo_PhiYg, MDTD_Demo_P = save_to_numpy_arrays()
        X = gen_syn_X(MDTD_Demo_PhiYg)  # numpy array of x * y * z
        num_modes = X.ndim
        lam, rho = gen_syn_lambda_rho(num_modes)  # list of size n
        phi_d = MDTD_Demo_Phi  # list of numpy arrays in form (1, n) where each atom corresponds to a dictionary.
        phi_type = ['not_ortho_dic', 'no_dic', 'no_dic']
        # first coordinate of each dictionary = shape of X
        P = MDTD_Demo_P  # Y values of shape of X
    else:
        X = X
        num_modes = X.ndim
        temp = lam
        lam = [temp] * num_modes if temp is not None else 0.000001
        rho = [val * 5 for val in lam]
        phi_d = np.empty((1, num_modes), dtype=object)

        for mode in range(num_modes):
            if mode == 0 or mode == 1:
                nested_dictionary = gen_gft_new(adj1, False)[0] if mode == 0 else gen_gft_new(adj2, False)[0]
            else:
                nested_dictionary = gen_rama(t=X.shape[2], max_period=24)
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
    PhiY = PhiYInit
    YPhi_Inner = YPhiInitInner
    Z = Y
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

        if (isinstance(mask, list) and len(mask) > 0) or (isinstance(mask, np.ndarray) and mask.any()):
            if isinstance(mask, list): mask = np.array(mask)
            # set D to reconstructed values and cast to double for mask indexing
            recon_t = tl.kruskal_to_tensor(
                (normalize_scaling.reshape((normalize_scaling.shape[1],)), [matrix for matrix in PhiY]))
            D = recon_t

            missing_mask, observed_mask = np.zeros(double_X.shape), np.ones(double_X.shape)

            # for idx in mask:
            #    nd_idx = nd_index(idx, double_X.shape)  # Convert index to correct tuple
            #    missing_mask[nd_idx] = 1
            nd_indices = np.unravel_index(mask, double_X.shape)
            missing_mask[nd_indices] = 1
            D = ((observed_mask - missing_mask) * (recon_t + lam[0] * double_X)) / 1 + lam[0]
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

    # [[S ⊡ Φ1Y1, Φ2Y2, Φ3Y3]
    return X, recon_t, PhiY


def mdtd_find_outlier(p_X, p_recon, p_count, p_slice) -> None:
    """
    Plots outliers based on magnitude from the residual of p_X-(p_recon).
    Args:
        p_X: Original temporal signal tensor
        p_recon: Reconstruction of p_X from S ⊡ Φ1Y1, Φ2Y2, Φ3Y3
        p_percentage: Top percentile of outliers that the user wishes to plot
        p_count: Number of subplots to display, one for each outlier in p_count. Maximum count of 10.
        p_slice: x/y/z plane to visualize tensor.
    """
    # TODO
    outlier_indices = []
    avg_magnitude_indices = []
    p_X, p_recon = np.abs(p_X), np.abs(p_recon)
    residual = p_X - p_recon
    p_count = min(p_count, 10)

    def arbitrary():
        for i in range(p_X.shape[2]):
            slice_flattened = residual[:, :, i].flatten()
            sorted_values = np.argsort(slice_flattened)[::-1]
            outlier_indices_fixed = sorted_values[:p_count]
            row_indices_fixed, col_indices_fixed = np.unravel_index(outlier_indices_fixed, p_X[:, :, i].shape)

            for r, c in zip(row_indices_fixed, col_indices_fixed):
                outlier_indices.append((r, c, i))

        pane_max_outlier_magnitude = {}
        for (r, c, pane) in outlier_indices:
            magnitude = p_X[r, c, pane]

            if pane in pane_max_outlier_magnitude:
                pane_max_outlier_magnitude[pane] = max(pane_max_outlier_magnitude[pane], magnitude)
            else:
                pane_max_outlier_magnitude[pane] = magnitude

        # Top 25 panes (arbitrary number) with the largest magnitude of outliers
        top_panes = sorted(pane_max_outlier_magnitude, key=pane_max_outlier_magnitude.get, reverse=True)[:25]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for pane_idx in top_panes:
            pane_outliers = [(r, c, z) for r, c, z in outlier_indices if z == pane_idx]
            magnitudes = [p_X[r, c, z] for r, c, z in pane_outliers]
            sizes = [(magnitude / max(magnitudes)) * 100 for magnitude in magnitudes]
            rows, cols, zs = zip(*pane_outliers)
            ax.scatter(rows, cols, zs=pane_idx, s=sizes, depthshade=True, label=f'Pane {pane_idx}')

        # Set labels and title
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xlabel('Row Index')
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_ylabel('Column Index')

        ax.set_zlabel('Pane Index')
        ax.set_title('Top 25 Panes with 10 Largest Outliers')

        # Set the ticks for the z-axis (pane index) to correspond to the indices of the panes
        ax.set_xticks([0, p_X.shape[0]])
        ax.set_yticks([0, p_X.shape[1]])
        ax.set_zticks([min(top_panes), max(top_panes)])

        # Adjust the view angle for better visualization
        # elev = height
        # azim = L to R
        ax.view_init(elev=30, azim=120)
        ax.legend(loc='center left', bbox_to_anchor=(1.10, 0.5), title='Pane Number')

        plt.show()

    def x_slice():
        average_magnitude_per_slice = np.mean(residual, axis=(1, 2))
        top_slices_indices = np.argsort(average_magnitude_per_slice)[::-1][:p_count]
        outlier_indices_sorted_by_pane = sorted(top_slices_indices)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        y_range = np.arange(p_X.shape[1])
        z_range = np.arange(p_X.shape[2])

        Y_mesh, Z_mesh = np.meshgrid(y_range, z_range)
        normalized_magnitudes = average_magnitude_per_slice[top_slices_indices] / np.max(
            average_magnitude_per_slice[top_slices_indices])

        for i, slice_index in enumerate(outlier_indices_sorted_by_pane):
            X_mesh = np.full(Y_mesh.shape, slice_index)
            alpha = normalized_magnitudes[i] * 0.9 + 0.1

            ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=alpha, label=f'X Slice #{slice_index}')

        # Set the labels and title
        ax.set_xlabel('Row Index')
        ax.set_ylabel('Column Index')
        ax.set_zlabel('Pane Index')
        ax.set_title('Top 10 Panes by Average Magnitude')

        # Adjust the legend and the view angle
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Pane Number')
        ax.view_init(elev=10, azim=-70)  # Adjust the view angle

        plt.show()

    def y_slice():
        average_magnitude_per_slice = np.mean(residual, axis=(0, 2))
        top_slices_indices = np.argsort(average_magnitude_per_slice)[::-1][:p_count]
        outlier_indices_sorted_by_pane = sorted(top_slices_indices)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        x_range = np.arange(p_X.shape[0])
        z_range = np.arange(p_X.shape[2])

        X_mesh, Z_mesh = np.meshgrid(x_range, z_range)
        normalized_magnitudes = average_magnitude_per_slice[top_slices_indices] / np.max(
            average_magnitude_per_slice[top_slices_indices])

        for i, slice_index in enumerate(outlier_indices_sorted_by_pane):
            Y_mesh = np.full(X_mesh.shape, slice_index)
            alpha = normalized_magnitudes[i] * 0.9 + 0.1

            ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=alpha, label=f'Y Slice #{slice_index}')

        # Set the labels and title
        ax.set_xlabel('Row Index')
        ax.set_ylabel('Column Index')
        ax.set_zlabel('Pane Index')
        ax.set_title('Top 10 Y Panes by Average Magnitude')

        # Adjust the legend and the view angle
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Pane Number')
        ax.view_init(elev=10, azim=-70)  # Adjust the view angle

        plt.show()

    def z_slice():
        average_magnitude_per_slice = np.mean(residual, axis=(0, 1))
        top_slices_indices = np.argsort(average_magnitude_per_slice)[::-1][:p_count]
        outlier_indices_sorted_by_pane = sorted(top_slices_indices)

        # Now plot the meshgrids for the selected panes
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Define the range for x and y
        x_range = np.arange(p_X.shape[0])
        y_range = np.arange(p_X.shape[1])

        # Create a meshgrid for x and y
        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
        normalized_magnitudes = average_magnitude_per_slice[top_slices_indices] / np.max(
            average_magnitude_per_slice[top_slices_indices])

        # Plot meshgrids for the top 10 panes by average magnitude
        for i, slice_index in enumerate(outlier_indices_sorted_by_pane):
            # Z position of the pane in the 3D plot
            Z_mesh = np.full(X_mesh.shape, slice_index)
            alpha = normalized_magnitudes[i] * 0.9 + 0.1  # Scale alpha between 0.2 and 1.0

            # Plot the meshgrid as a semi-transparent plane
            ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=alpha, label=f'Z Slice #{slice_index}')

        # Set the labels and title
        ax.set_xlabel('Row Index')
        ax.set_ylabel('Column Index')
        ax.set_zlabel('Pane Index')
        ax.set_title('Top 10 Z Panes by Average Magnitude')

        # Adjust the legend and the view angle
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Pane Number')
        ax.view_init(elev=10, azim=-70)  # Adjust the view angle

        plt.show()

    # Slice method
    if p_slice == "x":
        x_slice()
    elif p_slice == "y":
        y_slice()
    else:
        z_slice()


def mdtd_clustering(p_PhiY, n_clusters):
    fig, axes = plt.subplots(1, len(p_PhiY), figsize=(5*(1+len(p_PhiY)), 5))
    all_labels = []
    # Iterate through list of PhiY and perform KMeans on each
    for i, matrix in enumerate(p_PhiY):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(matrix)
        labels = kmeans.labels_
        all_labels.extend(labels)
        # Plot on figure
        axes[i].scatter(matrix[:, 0], matrix[:, 1], c=labels, cmap='viridis')
        axes[i].set_title(f'Matrix {i+1} Clusters')

    scatter = plt.scatter([], [], c=[], cmap='viridis')
    scatter.set_clim(min(all_labels), max(all_labels))
    # Define colorbar to fit RHS
    cbar = plt.colorbar(scatter, ax=axes.ravel().tolist(), pad=0.05)
    cbar.set_label('Cluster #')
    plt.show()

def mdtd_load_config():
    def build_adj_matrix(p_path, p_dim):
        adj_data = np.loadtxt(p_path, delimiter=',', dtype=int)
        rows, cols = adj_data[:, 0], adj_data[:, 1]
        return sp.csc_matrix((np.ones_like(rows - 1), (rows - 1, cols - 1)), shape=(p_dim, p_dim))

    with open('mdtd_config.json', 'r') as file:
        config = json.load(file)

    # Load X from a CSV file
    if config['X']:
        X = mdtd_format_csv_to_numpy(config['X'])
    if X.any():
        if config['adj-1']:
            adj_1 = build_adj_matrix(config["adj-1"], X.shape[0])
        if config['adj-2']:
            adj_2 = build_adj_matrix(config["adj-2"], X.shape[1])

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


def mdtd_format_numpy_to_csv(p_data):
    row_indices, column_indices, depth_indices = np.indices(p_data.shape)
    rows = row_indices.flatten()
    columns = column_indices.flatten()
    depths = depth_indices.flatten()
    values = p_data.flatten()
    df = pd.DataFrame({
        'Row': rows,
        'Column': columns,
        'Depth': depths,
        'Value': values
    })
    csv_file_path = '/Users/michaelpaglia/Documents/pyspady-personal/pyspady/mdtd_numpy_to_csv'
    df.to_csv(csv_file_path, index=False)
    csv_file_path


def mdtd_format_csv_to_numpy(p_data_csv):
    df = pd.read_csv(p_data_csv)
    max_row = df['Row'].max() + 1
    max_col = df['Column'].max() + 1
    max_depth = df['Depth'].max() + 1
    reconstructed_array = np.full((max_row, max_col, max_depth), np.nan, dtype=np.longdouble)
    rows = df['Row'].values
    cols = df['Column'].values
    depths = df['Depth'].values
    values = df['Value'].values
    reconstructed_array[rows, cols, depths] = values
    return reconstructed_array


if __name__ == '__main__':
    # mdtm_input_x, mdtm_input_adj, mdtm_input_mask, mdtm_input_count_nnz, mdtm_input_num_iters_check, mdtm_input_lam, mdtm_input_K, mdtm_input_epsilon = mdtm_load_config()
    mdtm_X, recon_X, phi_y = mdtd(is_syn=True, X=None, adj1=None, adj2=None, mask=[], count_nnz=0, num_iters_check=10,
                           lam=0.000001, K=10,
                           epsilon=1e-4)
    mdtd_clustering(phi_y, 10)
    # mdtm_find_outlier(mdtm_X, recon_X, 10)
