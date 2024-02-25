import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from tgsd import tgsd, config_run


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, K=7, iterations=100, lambda1=0.1, lambda2=0.1, lambda3=1, rho1=0.01, rho2=0.01):
        # Hyperparameters only, no data initialization
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
        Y, W = tgsd(X, self.psi, self.phi, self.mask, self.iterations, self.K, self.lambda1, self.lambda2, self.lambda3, self.rho1,
                    self.rho2, type="rand")
        self.Y_ = Y
        self.W_ = W
        return self

    def score(self, X, y=None):
        X = self.X
        residual = X - (self.psi @ self.Y_ @ self.W_ @ self.phi)
        return -np.linalg.norm(residual, ord='fro')/np.linalg.norm(X, ord='fro')


custom_encoder = CustomEncoder()

param_grid = {
    'K': [2, 4, 6, 8, 10],
    'lambda1': [.001, .01, .1, 1],
    'lambda2': [.001, .01, .1, 1],
    'lambda3': [.1, 1, 10]
}

grid_search = GridSearchCV(custom_encoder, param_grid)

grid_search.fit(custom_encoder.X)

print("Best parameters:", grid_search.best_params_)
print("Best score (negative Frobenius norm):", grid_search.best_score_)
