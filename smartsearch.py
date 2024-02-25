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
        X = self.X
        Y, W = tgsd(X, self.psi, self.phi, self.mask, self.iterations, self.K, self.lambda1, self.lambda2, self.lambda3,
                    self.rho1,
                    self.rho2, type="rand")
        self.Y_ = Y
        self.W_ = W
        return self

    def count_nonzero_with_epsilon(self, a, epsilon=1e-6):
        return np.sum(np.abs(a) >= epsilon)

    def calculate_coefficient_percentage(self):
        nonzero_elements = self.count_nonzero_with_epsilon(self.Y_) + self.count_nonzero_with_epsilon(self.W_)
        total_elements = self.Y_.size + self.W_.size
        return nonzero_elements / total_elements

    def calculate_residual_percentage(self):
        residual = self.X - (self.psi @ self.Y_ @ self.W_ @ self.phi)
        return np.linalg.norm(residual, ord='fro') / np.linalg.norm(self.X, ord='fro')


param_distributions = {
    'K': sp_randint(1, 11),
    'lambda1': uniform(0.001, 1),
    'lambda2': uniform(0.001, 1),
    'lambda3': uniform(0.1, 10)
}
param_sampler = ParameterSampler(param_distributions, n_iter=200, random_state=42)

best_coefficient_score = np.inf
best_residual_score = np.inf
best_coefficient_params = None
best_residual_params = None
best_params = None

residual_threshold = 0.8
coefficient_threshold = 0.001
found = False

for params in param_sampler:
    estimator = CustomEncoder(**params)
    estimator.fit(estimator.X)
    coefficient_percentage = estimator.calculate_coefficient_percentage()
    residual_percentage = estimator.calculate_residual_percentage()
    print(f"Coefficient%={coefficient_percentage} | Residual%={residual_percentage}")

    if residual_percentage < best_residual_score:
        best_residual_score = residual_percentage
        best_residual_params = params
    if coefficient_percentage < best_coefficient_score:
        best_coefficient_score = coefficient_percentage
        best_coefficient_params = params

    print(f"Best coefficient%={best_coefficient_score} | Best residual%={best_residual_score}")

    if residual_percentage < residual_threshold and coefficient_percentage < coefficient_threshold:
        print(f"Early stopping at residual score {residual_percentage} and coefficient score {coefficient_percentage} with parameters {params}"
        found = True
        break

if not found: print(f"After all possible iterations: Residual%={best_residual_score}; Best params {best_residual_params} | Coefficient%={best_coefficient_score}; Best params {best_coefficient_params}")
