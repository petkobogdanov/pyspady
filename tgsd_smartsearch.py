from sklearn.model_selection import ParameterSampler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import tgsd_home
import time

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config_path, demo, demo_X, demo_Phi, demo_Psi, demo_mask, coefficient_threshold, residual_threshold, optimizer_method, K=7, iterations=100, lambda1=0.1,
                 lambda2=0.1, lambda3=1, rho1=0.01, rho2=0.01):
        self.config_path = config_path
        self.coefficient_threshold = float(coefficient_threshold)
        self.residual_threshold = float(residual_threshold)
        self.best_coefficient_params = None
        self.best_residual_params = None
        self.best_overall_params = None
        self.best_coefficient_score = np.inf
        self.best_residual_score = np.inf
        self.demo = demo
        self.demo_X = demo_X
        self.demo_Psi = demo_Psi
        self.demo_Phi = demo_Phi
        self.demo_mask = demo_mask
        self.tgsd_driver = tgsd_home.TGSD_Home(config_path=self.config_path)
        self.optimizer_method = optimizer_method

        if self.demo:
            self.X = self.demo_X
            self.Psi_D = self.demo_Psi
            self.Phi_D = self.demo_Phi
            self.mask = self.demo_mask
        else:
            self.X = self.tgsd_driver.X
            self.Psi_D = self.tgsd_driver.Psi_D
            self.Phi_D = self.tgsd_driver.Phi_D
            self.mask = self.tgsd_driver.mask

        self.Y, self.W = None, None
        self.K = K
        self.iterations = iterations
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.rho1 = rho1
        self.rho2 = rho2

        start_time = time.time()
        self.tgsd_driver.tgsd(self.X, self.Psi_D, self.Phi_D, self.mask, self.iterations, self.K, self.lambda1, self.lambda2,
                                self.lambda3,
                                self.rho1,
                                self.rho2, type="rand", optimizer_method=self.optimizer_method)
        end_time = (time.time() - start_time) * 100 # n_iter = 100.

        # Hyperparameters to search using a uniform random distribution sampling
        self.param_distributions_mask = {
            'K': sp_randint(1, self.X.shape[0]+1),
            'lambda1': uniform(0.001, 5),
            'lambda2': uniform(0.001, 5),
            'lambda3': uniform(0.1, 10)
        }
        self.param_distributions_no_mask = {
            'K': sp_randint(1, self.X.shape[0]+1),
            'lambda1': uniform(0.001, 5),
            'lambda2': uniform(0.001, 5),
        }

        print(f"Preparing to run smart search. Estimated time to complete: {end_time // 60} minutes, {end_time % 60} seconds.")
    def fit(self, X, y=None):
        """
        Returns Y and W given certain permutation of hyperparameters
        Args:
            X: Input 2D temporal graph signal
        Returns:
            Y and W from TGSD
        """
        X = self.X
        Y, W = self.tgsd_driver.tgsd(X, self.Psi_D, self.Phi_D, self.mask, self.iterations, self.K, self.lambda1, self.lambda2,
                                self.lambda3,
                                self.rho1,
                                self.rho2, type="rand", optimizer_method=self.optimizer_method)
        self.Y_ = Y
        self.W_ = W
        return self

    def count_nonzero_with_epsilon(self, a, epsilon=1e-02):
        """
        Count number of non-zero, or close to non-zero, entries in some numpy array
        Args:
            a: Some numpy array
            epsilon: Threshold of what defines a "nonzero" entry
        Returns:
            Number of non-zero entries in some array a
        """
        return np.sum(np.abs(a) >= epsilon)

    def calculate_coefficient_percentage(self):
        """
        Calculate the percentage of coefficients that are nonzero in returned encoding matrices
        Returns:
            Percentage of non-zero coefficients from encoding matrices
        """
        nonzero_elements = self.count_nonzero_with_epsilon(self.Y_) + self.count_nonzero_with_epsilon(self.W_)
        total_elements = self.Y_.size + self.W_.size
        return nonzero_elements / total_elements

    def calculate_residual_percentage(self):
        """
        Calculate the percentage of how well the residual fits with the original input X
        Returns:
            Percentage of the residual compared to the original input X
        """
        residual = self.X - (self.Psi_D @ self.Y_ @ self.W_ @ self.Phi_D)
        return np.linalg.norm(residual, ord='fro') / np.linalg.norm(self.X, ord='fro')

    def get_best_residual_params(self):
        """
        Returns:
            Best combination of parameters for the residual scoring system
        """
        return self.best_residual_params

    def get_best_coefficient_params(self):
        """
        Returns:
            Best combination of parameters for the coefficient scoring system
        """
        return self.best_coefficient_params

    def get_best_overall_params(self):
        """
        Returns:
            Best overall parameters iff both the coefficient and residual scoring thresholds are satisfied
        """
        return self.best_overall_params

    def get_best_residual_score(self):
        """
        Returns:
            Best residual score
        """
        return self.best_residual_score

    def get_best_coefficient_score(self):
        """
        Returns:
            Best coefficient score
        """
        return self.best_coefficient_score

    def run_smart_search(self):
        """
        Given a parameter sampler, iterates through each parameter combination in the sampler and calculates
        the residual and coefficient scores. If either the residual or combination threshold is met, a new
        best threshold is stored. If both are met, the iteration breaks and the parameters are stored as optimal.
        """
        param_sampler = ParameterSampler(self.param_distributions_mask, n_iter=100, random_state=42) if len(
            self.mask) > 0 else \
            ParameterSampler(self.param_distributions_no_mask, n_iter=100, random_state=42)
        found = False
        for params in param_sampler:
            self.set_params(**params)
            self.fit(self.X)
            # Calculate coefficient and residual percentages
            coefficient_percentage = self.calculate_coefficient_percentage()
            residual_percentage = self.calculate_residual_percentage()
            print(f"Coefficient%={coefficient_percentage} | Residual%={residual_percentage}")
            # Update best scores and parameters if better scores are found
            if residual_percentage < self.best_residual_score:
                self.best_residual_score = residual_percentage
                self.best_residual_params = params
            if coefficient_percentage < self.best_coefficient_score:
                self.best_coefficient_score = coefficient_percentage
                self.best_coefficient_params = params

            # Check for early stopping conditions
            if residual_percentage < self.residual_threshold and coefficient_percentage < self.coefficient_threshold:
                print(
                    f"Early stopping at residual score {residual_percentage} and coefficient score {coefficient_percentage} with parameters {params}")
                print("Running best parameters on your input data...")
                self.best_overall_params = params
                found = True
                break

        if not found:
            print(
                f"After all possible iterations: Residual%={self.best_residual_score}; Best params {self.best_residual_params} | Coefficient%={self.best_coefficient_score}; Best params {self.best_coefficient_params}")

    def get_Y_W(self):
        """
        Given an optimal set of parameters, computes the encoding matrices Y and W
        Returns:
            Encoding matrices Y and W.
        """
        K = self.best_overall_params['K']
        lambda_1, lambda_2, lambda_3 = self.best_overall_params['lambda1'], self.best_overall_params['lambda2'], self.best_overall_params['lambda3']
        self.Y, self.W = self.tgsd_driver.tgsd(self.X, self.Psi_D, self.Phi_D, self.mask, k=K, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, optimizer_method=self.optimizer_method)
        return self.Y, self.W
