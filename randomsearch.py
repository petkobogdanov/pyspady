import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from tgsd import tgsd, config_run
from scipy.stats import randint as sp_randint
from scipy.stats import uniform


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, K=7, iterations=100, lambda1=0.1, lambda2=0.1, lambda3=1, rho1=0.01, rho2=0.01, residual_threshold=0.05, coefficient_threshold=0.1):
        # Initialization with user thresholds):
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
        self.residual_threshold = residual_threshold
        self.coefficient_threshold = coefficient_threshold

    def fit(self, X, y=None):
        X = self.X
        Y, W = tgsd(X, self.psi, self.phi, self.mask, self.iterations, self.K, self.lambda1, self.lambda2, self.lambda3, self.rho1,
                    self.rho2, type="rand")
        self.Y_ = Y
        self.W_ = W
        return self

    def score(self, X, y=None):
        X = self.X
        residual = X - (self.psi @ self.Y_ @ self.W_ @ self.phi)
        residual_percentage = -np.linalg.norm(residual, ord='fro')/np.linalg.norm(X, ord='fro')
        #coefficient_percentage = (np.count_nonzero(self.Y_) + np.count_nonzero(self.W_)) / (self.Y_.size + self.W_.size)
        return residual_percentage


custom_encoder = CustomEncoder()
param_distributions = {
    'K': sp_randint(1, 11),
    'lambda1': uniform(0.001, 1),
    'lambda2': uniform(0.001, 1),
    'lambda3': uniform(0.1, 10)
}
random_search = RandomizedSearchCV(custom_encoder, param_distributions, n_iter=100, cv=5, random_state=42)
random_search.fit(custom_encoder.X)
print("Best parameters:", random_search.best_params_)
print("Best score (negative Frobenius norm):", random_search.best_score_)
