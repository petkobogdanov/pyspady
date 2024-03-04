import os
import math
import numpy as np
import scipy.sparse as sp
from Y_unittest import TestYConversion
from W_unittest import TestWConversion
import json
import random
import dictionary_generation


class TGSD_Home:
    def __init__(self, config_path):
        self.config_path = config_path
        self.X, self.Psi_D, self.Phi_D, self.mask = self.config_run(config_path=f"{self.config_path}")
        self.Y, self.W = None, None

    def tgsd(self, X, psi_d, phi_d, mask,
             iterations: int = 100, k: int = 7, lambda_1: int = 0.1, lambda_2: int = 0.1, lambda_3: int = 1,
             rho_1: int = 0.01, rho_2: int = 0.01, type: str = "rand"):
        """
        Decomposes a temporal graph signal as a product of two fixed dictionaries and two corresponding sparse encoding matrices
        Args:
            X: Temporal graph signal input
            psi_d: Some graph dictionary, Ψ
            phi_d: Some time series dictionary,
            mask: List of linear indices to disguise
            iterations: Number of iterations to run on algorithm
            k: Rank of the encoding matrices
            lambda_1: Some sparsity regularization parameter
            lambda_2: Some sparsity regularization parameter
            lambda_3: Some sparsity regularization parameter
            rho_1: Some penalty parameter
            rho_2: Some penalty parameter
            type: Row, column, or pred mask

        Returns:
            Sparse encoding matrices Y and W
        """
        self.X = X
        self.Psi_D = psi_d
        self.Phi_D = phi_d
        self.mask = mask

        def is_orthonormal(p_psi_or_phi):
            """
            Determines if graph or time series dictionary is orthonormal
            Args:
                p_psi_or_phi: Specified graph or time series dictionary
            Returns:
                True if orthonormal, false otherwise
            """
            return np.allclose(np.dot(p_psi_or_phi.T.conj(), p_psi_or_phi), np.eye(p_psi_or_phi.shape[1])) and \
                   np.allclose(np.linalg.norm(p_psi_or_phi, axis=0), np.ones(p_psi_or_phi.shape[1]))

        def update_d(p_P, p_X, p_mask, p_lambda_3):
            """
            Learns D via P, X, mask, and lambda 3
            Args:
                p_P: ΨYWΦ
                p_X: Some temporal graph signal
                p_mask: Indices represent masked indices in D
                p_lambda_3: Some sparsity regularization parameter, λ3
            Returns:
                D = (P +𝜆3Ω ⊙ X) ⊘ (I +𝜆3Ω)
            """
            p_mask, missing_mask, observed_mask = p_mask - 1, np.zeros(p_P.shape), np.ones(p_P.shape)
            missing_mask[p_mask % p_P.shape[0], p_mask // p_P.shape[0]] = 1
            return (p_P.astype(np.complex128) + (p_lambda_3 * (observed_mask - missing_mask)) * p_X.astype(
                np.longdouble)) / (
                           1 + (p_lambda_3 * (observed_mask - missing_mask)))

        def hard_update_y(p_sigma, p_W, p_X, p_Z, p_psi, p_phi, p_lambda_1, p_Q_1, p_gamma_1, p_rho_1):
            """
            Learn Y based on E1: Y = Q1E1QT2.
            Args:
                p_sigma: Some identity matrix, σ
                p_W: Some encoding matrix, W
                p_X: Some temporal graph signal, X
                p_Z: Some intermediate variable, Z
                p_psi: Some dictionary of non-orthogonal atoms, Ψ
                p_phi: Some time dictionary of atoms, Φ
                p_lambda_1: Some diagonal non-negative eigenvalue matrix, λ1
                p_Q_1: Some orthonormal eigenvector matrix, Q1
                p_gamma_1: Some Lagrangian multiplier, Γ1
                p_rho_1: Some penalty parameter, ρ1

            Returns:
                Y = Q1E1QT2
            """
            _B = p_sigma.astype(np.longdouble) @ p_W.astype(np.complex128) @ p_phi.astype(np.complex128)
            _lambda_2, Q_2 = np.linalg.eigh((_B @ _B.conj().T), UPLO='U')
            _lambda_2 = np.diag(_lambda_2)
            # Pi=2*Psi'*X*B'+rho_1*Z+Gamma_1;
            Pi = 2 * p_psi.astype(np.longdouble).T @ p_X.astype(np.complex128) @ (
                _B.astype(np.complex128)).conj().T + p_rho_1 * p_Z.astype(np.complex128) + p_gamma_1

            QPiQ = p_Q_1.astype(np.longdouble).T @ Pi.astype(np.complex128) @ Q_2.astype(np.complex128)
            diagonal_lambda_1, diagonal_lambda_2 = np.diag(p_lambda_1)[:, None], (np.diag(_lambda_2)[:, None]).T
            temp0 = 2 * diagonal_lambda_1.astype(np.longdouble) @ diagonal_lambda_2.astype(np.longdouble) + rho_2
            E = QPiQ / temp0
            return p_Q_1.astype(np.longdouble) @ E.astype(np.complex128) @ Q_2.astype(np.complex128).conj().T

        def hard_update_w(p_sigma, p_V, p_X, p_Y, p_psi, p_phi, p_lambda_4, p_Q_4, p_gamma_2, p_rho_2):
            """
            Learns W where W = Q3E2QT4, where E2(i, j) = [QT3 Π 2Q4]i,j/2[Λ4]ii[Λ3]jj+p2 and
            (Q3,Λ3 and (Q4,Λ4) are the (eigenvector, eigenvalue) matrices of
            ATA and ΦΦT , respectively.
            Args:
                p_sigma: Some identity matrix, σ
                p_V: Some intermediate variable, V
                p_X: Some temporal graph signal, X
                p_Y: Some encoding matrix, Y
                p_psi: Some dictionary of atoms, Ψ
                p_phi: Some non-orthogonal time dictionary of atoms, Φ
                p_lambda_4: Some diagonal non-negative eigenvalue matrix, λ4
                p_Q_4: Some orthonormal eigenvector matrix, Q4
                p_gamma_2: Some Lagrangian multiplier, Γ2
                p_rho_2: Some penalty parameter, ρ2

            Returns:
                W = Q3E2QT4
            """
            _A = p_psi.astype(np.longdouble) @ p_Y.astype(np.complex128) @ p_sigma.astype(np.longdouble)
            _lambda_3, Q_3 = np.linalg.eigh((_A.conj().T @ _A), UPLO='U')
            _lambda_3 = np.diag(_lambda_3)
            # Pi=2*A'*X*Phi'+rho_2*V+Gamma_2;
            Pi = 2 * (_A.astype(np.complex128)).conj().T @ p_X.astype(np.complex128) @ (
                p_phi.astype(np.complex128)).conj().T + p_rho_2 * p_V + p_gamma_2
            QPiQ = Q_3.astype(np.complex128).conj().T @ Pi @ p_Q_4.astype(np.complex128)

            diagonal_lambda_3, diagonal_lambda_4 = np.diag(_lambda_3)[:, None], (np.diag(p_lambda_4)[:, None]).T
            temp0 = 2 * diagonal_lambda_3.astype(np.longdouble) @ diagonal_lambda_4.astype(np.longdouble) + rho_2
            E = QPiQ / temp0
            return Q_3.astype(np.complex128) @ E.astype(np.complex128) @ Q_4.astype(np.complex128).conj().T

        def get_object(p_mask, p_D, p_X, p_phi, p_psi, p_Y, p_sigma, p_W, p_lambda_1, p_lambda_2, p_lambda_3):
            """
            Returns a new object represented by equation X−YΨWΦ‖
            Args:
                p_mask: Some specified indexed mask
                p_D: Some intermediate variable, D
                p_X: Some temporal graph signal, X
                p_phi: Some time dictionary of atoms Phi
                p_psi: Some dictionary of atoms, Psi
                p_Y: Some encoding matrix Y
                p_sigma: Some identity matrix, σ
                p_W: Some encoding matrix W
                p_lambda_1: Some sparsity regularization parameter, λ1
                p_lambda_2: Some sparsity regularization parameter, λ2
                p_lambda_3: Some sparsity regularization parameter, λ3

            Returns:
                New object represented by X−YΨWΦ‖
            """
            # temp=X(:)-D(:);
            # term3=norm(temp(setdiff(1:end,mask)));
            p_D, p_phi, p_Y, p_W = p_D.astype(np.complex128), p_phi.astype(np.complex128), p_Y.astype(
                np.complex128), p_W.astype(np.complex128)
            p_sigma, p_X, p_psi = p_sigma.astype(np.longdouble), p_X.astype(np.longdouble), p_psi.astype(np.longdouble)

            p_mask, missing_mask, observed_mask = p_mask - 1, np.zeros(p_X.shape), np.ones(p_X.shape)
            missing_mask[p_mask % p_X.shape[0], p_mask // p_X.shape[0]] = 1
            term_3 = np.linalg.norm((observed_mask - missing_mask) * (p_X - p_D), ord=2)
            # obj_new = norm(D-Psi*Y*Sigma*W*Phi)+lambda_1*norm(Y,1)+lambda_2*norm(W,1)+lambda_3*term3;
            return np.linalg.norm(p_D - p_psi @ p_Y @ p_sigma @ p_W @ p_phi, ord=2) \
                   + p_lambda_1 * np.linalg.norm(np.abs(p_Y), ord=1) + p_lambda_2 * np.linalg.norm(np.abs(p_W),
                                                                                                   ord=1) + p_lambda_3 * term_3

        _, t = X.shape
        hold, Y1 = psi_d.shape
        p, t = phi_d.shape
        self.W = np.zeros((k, p))
        self.W[0, 0] = .000001
        if (not is_orthonormal(phi_d) and not is_orthonormal(psi_d)) or (
                is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
            self.Y = np.eye(Y1, k)
        else:
            self.Y = np.zeros((Y1, k))
        sigma = np.eye(k, k)
        sigmaR = np.eye(sigma.shape[0], sigma.shape[1])
        V, Z = 0, self.Y
        gamma_1, gamma_2 = self.Y, self.W
        obj_old, objs = 0, []
        I_Y = np.eye((self.W @ phi_d @ (self.W @ phi_d).T).shape[0])
        I_W = np.eye(((psi_d @ self.Y).T @ psi_d @ self.Y).shape[0])
        # [Q_1,Lam_1]=eig(Psi'*Psi);
        lam_1, Q_1 = np.linalg.eigh((psi_d.T @ psi_d), UPLO='U')
        lam_4, Q_4 = np.linalg.eigh((phi_d.astype(np.complex128) @ phi_d.astype(np.complex128).conj().T), UPLO='U')
        lam_1, lam_4 = np.diag(lam_1), np.diag(lam_4)
        XPhiT = X.astype(np.complex128) @ phi_d.astype(np.complex128).conj().T
        PsiTX = psi_d.astype(np.longdouble).T @ X.astype(np.complex128)

        if mask.any():
            if type == "row":  # row-mask
                n, m = X.shape
                temp2 = np.ones((n, m))
                temp2[mask - 1 % X[0], :] = 0
                self.mask = np.argwhere(temp2 == 0)
            elif type == "col" or type == "pred":  # column or pred mask
                n, m = X.shape
                temp2 = np.ones((n, m))
                temp2[:, mask - 1] = 0
                self.mask = np.argwhere(temp2 == 0)

        # plt.figure()

        for i in range(1, 1 + iterations):
            P = (psi_d.astype(np.longdouble) @ self.Y.astype(np.complex128) @ self.W.astype(
                np.complex128) @ phi_d.astype(
                np.complex128))
            D = update_d(P, X, mask, lambda_3)
            #     B=Sigma*W*Phi;
            #     Y=(2*Psi'*D*B'+rho_1*Z+Gamma_1)*inv(2*(B*B')+rho_1*I_y+exp(-15));
            B = sigma.astype(np.longdouble) @ self.W.astype(np.complex128) @ phi_d.astype(np.complex128)
            if mask.any():
                # Both are orth OR Psi is orth, Phi is not orth
                if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                        is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                    self.Y = (2 * psi_d.conj().T.astype(np.longdouble) @ D.astype(
                        np.complex128) @ B.conj().T + rho_1 * Z.astype(
                        np.complex128) + gamma_1.astype(np.complex128)) @ np.linalg.pinv(
                        2 * (B @ B.conj().T) + rho_1 * I_Y + math.exp(-15)).astype(np.complex128)
                # Phi is orth, Psi is not orth OR Psi is not orth, Phi is not orth
                elif (is_orthonormal(phi_d) and not is_orthonormal(psi_d)) or (
                        not is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                    self.Y = hard_update_y(sigma, self.W, D, Z, psi_d, phi_d, lam_1, Q_1, gamma_1, rho_1)
            else:
                # Both are orth OR Psi is orth, Phi is not orth
                if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                        is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                    # Y=(2*PsiTX*B'+rho_1*Z+Gamma_1)*inv(2*(B*B')+rho_1*I_y+exp(-15));
                    self.Y = (2 * PsiTX.astype(np.complex128) @ (B.astype(np.complex128)).conj().T + rho_1 * Z.astype(
                        np.complex128) + gamma_1) \
                             @ np.linalg.pinv(2 * (B @ B.conj().T) + rho_1 * I_Y + math.exp(-15)).astype(np.complex128)
                elif (is_orthonormal(phi_d) and not is_orthonormal(psi_d)) or (
                        not is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
                    self.Y = hard_update_y(sigma, self.W, D, psi_d, phi_d, lam_1, Q_1, gamma_1, rho_1)

            test_instance = TestYConversion()
            # ans_y = test_instance.test_y_complex_conversion(i, Y)
            # Update Z:
            # h = Y-gamma_1 / rho_1
            # Z = sign(h).*max(abs(h)-lambda_1/rho_1, 0)
            h = (self.Y - (gamma_1.astype(np.complex128) / rho_1)).astype(np.complex128)
            Z = (np.sign(h) * np.maximum(np.abs(h) - (lambda_1 / rho_1), 0)).astype(np.complex128)
            # A = psi*Y*sigma
            A = psi_d.astype(np.longdouble) @ self.Y.astype(np.complex128) @ sigma.astype(np.longdouble)
            if mask.any():
                # Both are orthonormal OR Psi is orth, Phi is not
                if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                        is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
                    # W = inv(2*(A')*A + I_W*rho_2) * (2*A'*D*phi'+rho_2*V+gamma_2)
                    self.W = np.linalg.pinv(2 * A.conj().T @ A + I_W * rho_2).astype(np.complex128) @ (
                            2 * A.conj().T @ D.astype(np.complex128) @ phi_d.conj().T + rho_2 * V + gamma_2)
                # Psi is orth, Phi is not orth OR Psi is not orth, Phi is not orth
                elif (is_orthonormal(psi_d) and not is_orthonormal(phi_d)) or (
                        not is_orthonormal(psi_d) and not is_orthonormal(phi_d)):
                    self.W = hard_update_w(sigma, V, D, self.Y, psi_d.astype(np.longdouble),
                                           phi_d.astype(np.complex128), lam_4, Q_4,
                                           gamma_2, rho_2)
            else:
                # Both are orth OR Phi is orth, Psi is not
                if (is_orthonormal(phi_d) and is_orthonormal(psi_d)) or (
                        is_orthonormal(phi_d) and not is_orthonormal(psi_d)):
                    # W=inv(2*(A')*A + I_w*rho_2)*(2*A'*XPhiT+rho_2*V+ Gamma_2);
                    self.W = np.linalg.pinv(2 * A.conj().T @ A + I_W * rho_2).astype(np.complex128) @ (
                            2 * A.conj().T @ XPhiT.astype(np.complex128) + rho_2 * V + gamma_2)
                # Psi is orth, Phi is not
                elif is_orthonormal(psi_d) and not is_orthonormal(phi_d):
                    sigma = sigma @ sigmaR
                    self.W = hard_update_w(sigma, V, D, self.Y, psi_d.astype(np.longdouble),
                                           phi_d.astype(np.complex128), lam_4, Q_4,
                                           gamma_2, rho_2)
                elif not is_orthonormal(psi_d) and not is_orthonormal(phi_d):
                    self.W = hard_update_w(sigma, V, D, self.Y, psi_d.astype(np.longdouble),
                                           phi_d.astype(np.complex128), lam_4, Q_4,
                                           gamma_2, rho_2)

            test_instance_w = TestWConversion()
            # ans_w = test_instance_w.test_w_complex_conversion(i, W)
            # Update V:
            # h= W-Gamma_2/rho_2;
            # V = sign(h).*max(abs(h)-lambda_2/rho_2,0);
            h = self.W - (gamma_2.astype(np.complex128) / rho_2)
            V = (np.sign(h) * np.maximum(np.abs(h) - (lambda_2 / rho_2), 0))

            gamma_1, gamma_2 = gamma_1 + rho_1 * (Z - self.Y), gamma_2 + rho_2 * (V - self.W)
            rho_1, rho_2 = min(rho_1 * 1.1, 1e5), min(rho_2 * 1.1, 1e5)

            # Stop condition
            if i % 25 == 0:
                obj_new = get_object(mask, D, X, phi_d, psi_d, self.Y, sigma, self.W, lambda_1, lambda_2, lambda_3)
                objs = [objs, obj_new]
                residual = abs(obj_old - obj_new)
                print(f"obj-{i}={obj_new}, residual-{i}={residual}")
                if residual < 1e-6:
                    break
                else:
                    obj_old = obj_new
        return self.Y, self.W

    def config_run(self, config_path: str = "config.json"):
        # Try to open the config file
        try:
            with open(config_path) as file:
                config: json = json.load(file)
        except FileNotFoundError:
            raise Exception(f"Config file '{config_path}' not found")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON format in '{config_path}'")
        except Exception as e:
            raise Exception(f"Error loading config file: {e}")

        # Validate the mandatory keys
        if not ("psi" in config):
            raise Exception("Config must contain the 'psi' key")
        if not ("phi" in config):
            raise Exception("Config must contain the 'phi' key")
        if not ("x" in config):
            raise Exception("Config must contain the 'x' key")
        if not ("mask_mode" in config):
            raise Exception("Config must contain the 'mask_mode' key")
        if not ("mask_percent" in config):
            raise Exception("Config must contain the 'mask_percent' key")
        if not ("first_x_dimension" in config):
            raise Exception("Config must contain the 'first_x_dimension' key")
        if not ("second_x_dimension" in config):
            raise Exception("Config must contain the 'second_x_dimension' key")

        # Validate the first and second dimensions of x
        first_x_dimension: int = config["first_x_dimension"]
        second_x_dimension: int = config["second_x_dimension"]
        if not (isinstance(first_x_dimension, int)):
            raise Exception(f"Key 'first_x_dimension', {first_x_dimension}, is invalid. Please enter a valid int")
        if not (isinstance(second_x_dimension, int)):
            raise Exception(f"Key 'second_x_dimension', {second_x_dimension}, is invalid. Please enter a valid int")

        psi: str = str(config["psi"]).lower()
        phi: str = str(config["phi"]).lower()

        # Validate the runnability of the instance
        if psi != "gft" and phi != "gft":
            raise Exception("At least one of PSI or PHI must be 'gft'")

        save_flag: bool = False
        load_flag: bool = False

        # Check if the save flag in the config is enabled and validate the input
        if "save_flag" in config:
            if not isinstance(config["save_flag"], bool):
                raise Exception("Invalid 'save_flag', must be a boolean")
            else:
                save_flag = config["save_flag"]

        # Check if the load flag in the config is enabled and validate the input
        if "load_flag" in config:
            if not isinstance(config["load_flag"], bool):
                raise Exception("Invalid 'load_flag', must be a boolean")
            else:
                load_flag = config["load_flag"]

        # Try to load the data
        try:
            data: np.ndarray[any] = np.genfromtxt(config["x"], delimiter=',')
        except Exception as e:
            raise Exception(f"Error loading data from '{config['x']}': {e}")

        match str(config["psi"]).lower():
            case "ram":
                # psi_d = gen_rama(400, 10)
                psi_d = dictionary_generation.GenerateDictionary.gen_rama(data.shape[1], 10)
                pass
            case "gft":
                # Attempt to load adj_list
                try:
                    adj_data: np.ndarray[any] = np.loadtxt(config["adj_path"], delimiter=',', dtype=int)
                except Exception as e:
                    raise Exception(f"Error loading adj_list data from '{config['adj_path']}': {e}")
                # Validate the adjacency matrix's dimension
                if not ("adj_square_dimension" in config):
                    raise Exception("PSI's dictionary, GFT, requires 'adj_square_dimension' key")
                adj_square_dimension: int = config["adj_square_dimension"]
                if not (isinstance(adj_square_dimension, int)):
                    raise Exception(
                        f"Key, 'adj_square_dimension', {adj_square_dimension} is invalid. Please enter a valid int")

                rows, cols = adj_data[:, 0], adj_data[:, 1]
                sparse_adj_mtx = sp.csc_matrix((np.ones_like(rows), (rows, cols)),
                                               shape=(adj_square_dimension, adj_square_dimension))
                gft = dictionary_generation.GenerateDictionary.gen_gft_new(sparse_adj_mtx, False)
                psi_d = gft[0]  # eigenvectors
                pass
            case "dft":
                pass
                # psi_d = gen_dft(200)
            case _:
                raise Exception(f"PSI's dictionary, {config['psi']}, is invalid")

        match str(config["phi"]).lower():
            case "ram":
                # phi_d = gen_rama(400, 10)
                pass
            case "gft":
                # Validate the adjacency matrix's dimension
                if not ("adj_square_dimension" in config):
                    raise Exception("PHI's dictionary, GFT, requires 'adj_square_dimension' key")
                adj_square_dimension: int = config["adj_square_dimension"]
                if not (isinstance(adj_square_dimension, int)):
                    raise Exception(
                        f"Key, 'adj_square_dimension', {adj_square_dimension} is invalid. Please enter a valid int")
                pass
            case "dft":
                phi_d = dictionary_generation.GenerateDictionary.gen_dft(data.shape[1])
                pass
            case _:
                raise Exception(f"PHI's dictionary, {config['phi']}, is invalid")

        # Validate the mask percent
        mask_percent: int = config["mask_percent"]
        if not (isinstance(mask_percent, int) or (mask_percent < 0 or mask_percent > 100)):
            raise Exception(f"{mask_percent} is invalid. Please enter a valid percent")

        # If the load flag is enabled load from file
        if (load_flag):
            # Retrieve the correct path
            load_path: str = config["load_path"] if "load_path" in config else "save.match"
            # Try to load the data
            try:
                mask_data: np.ndarray[any] = np.loadtxt(load_path, dtype=float)
            except FileNotFoundError:
                raise Exception(f"Load path '{load_path}' does not exist")
        # If the load flag is not enabled check the mask mode
        else:
            # Validate and read the mask mode
            match str(config["mask_mode"]).lower():
                case "lin":
                    mask_data: np.ndarray[any] = np.linspace(1, round(mask_percent / 100 * data.size),
                                                             round(mask_percent / 100 * data.size), dtype=np.uint16)
                case "rand":
                    mask_data: np.ndarray[any] = np.array(
                        random.sample(range(1, data.size), round(mask_percent / 100 * data.size)))
                case "path":
                    if not ("mask_path" in config):
                        raise Exception("Config must contain the 'mask_path' key when mask_mode = is path")
                    # Attempt to load mask data
                    try:
                        mask_data: np.ndarray[any] = np.genfromtxt(config["mask_path"], delimiter=',', ndmin=2,
                                                                   dtype=np.uint16)
                    except Exception as e:
                        raise Exception(f"Error loading mask data from '{config['mask_path']}': {e}")
                case _:
                    raise Exception(f"Invalid 'mask_mode': {config['mask_mode']}")

        # If the save flag is enabled save to file
        if (save_flag):
            # Retrieve the the correct path
            save_path: str = config["save_path"] if "save_path" in config else "save.match"
            # Insure that data is not overwritten without user consent
            if (os.path.exists(save_path)):
                # If user permission is given try to write the data
                if ("override" in config and config["override"]):
                    try:
                        np.savetxt(save_path, mask_data)
                    except Exception as e:
                        raise Exception(f"Error saving data: {e}")
                # If user permission is not granted raise an exception
                else:
                    raise Exception(f"{save_path} already exists. Enable override to override the saved data.")
            # If the path does not already exist try to write the data
            else:
                try:
                    np.savetxt(save_path, mask_data)
                except Exception as e:
                    raise Exception(f"Error saving data: {e}")

        return data, psi_d, phi_d, mask_data
