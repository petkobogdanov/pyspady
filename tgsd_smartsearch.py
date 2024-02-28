from sklearn.model_selection import ParameterSampler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tgsd import tgsd, config_run
from scipy.stats import randint as sp_randint
from scipy.stats import uniform


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, K=7, iterations=100, lambda1=0.1, lambda2=0.1, lambda3=1, rho1=0.01, rho2=0.01):
        X, psi, phi, mask = config_run("config.json")
        self.X = X
        self.psi = psi
        self.phi = phi
        self.mask = mask
        self.K = K
        self.iterations = iterations
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.rho1 = rho1
        self.rho2 = rho2

    def fit(self, X, y=None):
        """
        Returns Y and W given certain permutation of hyperparameters
        Args:
            X: Input 2D temporal graph signal
        Returns:
            Y and W from TGSD
        """
        X = self.X
        Y, W = tgsd(X, self.psi, self.phi, self.mask, self.iterations, self.K, self.lambda1, self.lambda2, self.lambda3,
                    self.rho1,
                    self.rho2, type="rand")
        self.Y_ = Y
        self.W_ = W
        return self

    def count_nonzero_with_epsilon(self, a, epsilon=1e-6):
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
        residual = self.X - (self.psi @ self.Y_ @ self.W_ @ self.phi)
        return np.linalg.norm(residual, ord='fro') / np.linalg.norm(self.X, ord='fro')


# Hyperparameters to search using a uniform random distribution sampling
param_distributions = {
    'K': sp_randint(1, 11),
    'lambda1': uniform(0.001, 1),
    'lambda2': uniform(0.001, 1),
    'lambda3': uniform(0.1, 10)
}
# n_iters denotes how many times a permutation is chosen across cv #folds
param_sampler = ParameterSampler(param_distributions, n_iter=200, random_state=42)
# Custom break mechanism
# Define best coefficient% and residual% scores as some very large number
best_coefficient_score = np.inf
best_residual_score = np.inf
best_coefficient_params = None
best_residual_params = None
best_params = None
# Meant to be user-input
residual_threshold = 0.8
coefficient_threshold = 0.001
# Whether we found a valid permutation of parameters before breaking after n_iter iterations
found = False
# Iterate through each parameter permutation
for params in param_sampler:
    estimator = CustomEncoder(**params)
    estimator.fit(estimator.X)
    # Calculate coefficient and residual percentages
    coefficient_percentage = estimator.calculate_coefficient_percentage()
    residual_percentage = estimator.calculate_residual_percentage()
    print(f"Coefficient%={coefficient_percentage} | Residual%={residual_percentage}")
    # Define new residual percentage and best parameters for this
    if residual_percentage < best_residual_score:
        best_residual_score = residual_percentage
        best_residual_params = params
    # Define new coefficient percentage and best parameters for this
    if coefficient_percentage < best_coefficient_score:
        best_coefficient_score = coefficient_percentage
        best_coefficient_params = params

    print(f"Best coefficient%={best_coefficient_score} | Best residual%={best_residual_score}")

    # Valid permutation found given user input for residual and coefficient thresholds
    if residual_percentage < residual_threshold and coefficient_percentage < coefficient_threshold:
        print(f"Early stopping at residual score {residual_percentage} and coefficient score {coefficient_percentage} with parameters {params}")
        found = True
        break
# Valid permutation was not found, so print out the best scores and parameters found for each percentage
if not found: print(f"After all possible iterations: Residual%={best_residual_score}; Best params {best_residual_params} | Coefficient%={best_coefficient_score}; Best params {best_coefficient_params}")
