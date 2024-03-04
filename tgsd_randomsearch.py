import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from tgsd_home import tgsd, config_run
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
        Y, W = tgsd(X, self.psi, self.phi, self.mask, self.iterations, self.K, self.lambda1, self.lambda2, self.lambda3, self.rho1,
                    self.rho2, type="rand")
        self.Y_ = Y
        self.W_ = W
        return self

    def score(self, X, y=None):
        """
        Returns a residual% score of how well reconstructed X fits original X
        Args:
            X: Input 2D temporal graph signal
        Returns:
            Score of the Frobenius norm. Score is denoted as negative because GridSearch chooses maximum value
            and, in this case, a smaller score is considered better.
        """
        X = self.X
        residual = X - (self.psi @ self.Y_ @ self.W_ @ self.phi)
        residual_percentage = -np.linalg.norm(residual, ord='fro')/np.linalg.norm(X, ord='fro')
        return residual_percentage


# Define custom encoder
custom_encoder = CustomEncoder()
# Hyperparameters to search using a uniform random distribution sampling
param_distributions_mask = {
    'K': sp_randint(1, 11),
    'lambda1': uniform(0.001, 1),
    'lambda2': uniform(0.001, 1),
    'lambda3': uniform(0.1, 10)
}
param_distributions_no_mask = {
    'K': sp_randint(1, 11),
    'lambda1': uniform(0.001, 1),
    'lambda2': uniform(0.001, 1),
}
# n_iters denotes how many times a permutation is chosen across cv #folds
random_search = RandomizedSearchCV(custom_encoder, param_distributions_mask, n_iter=100, cv=5, random_state=42) if len(custom_encoder.mask) > 0 else\
    RandomizedSearchCV(custom_encoder, param_distributions_no_mask, n_iter=100, cv=5, random_state=42)
# Fit parameters to custom encoder
random_search.fit(custom_encoder.X)
print("Best parameters:", random_search.best_params_)
print("Best score (negative Frobenius norm):", random_search.best_score_)
