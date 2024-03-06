import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
import tgsd_home

TGSD_Driver = tgsd_home.TGSD_Home


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config_file, K=7, iterations=100, lambda1=0.1, lambda2=0.1, lambda3=1, rho1=0.01, rho2=0.01):
        X, psi, phi, mask = TGSD_Driver.config_run(config_file)
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
        self.best_params = None
        self.best_score = None
        self.param_grid_mask = {
            'K': [2, 4, 6, 8, 10],
            'lambda1': [.001, .01, .1, 1],
            'lambda2': [.001, .01, .1, 1],
            'lambda3': [.1, 1, 10]
        }
        self.param_grid_no_mask = {
            'K': [2, 4, 6, 8, 10],
            'lambda1': [.001, .01, .1, 1],
            'lambda2': [.001, .01, .1, 1],
        }

    def fit(self, X, y=None):
        """
        Returns Y and W given certain permutation of hyperparameters
        Args:
            X: Input 2D temporal graph signal
        Returns:
            Y and W from TGSD
        """
        X = self.X
        Y, W = TGSD_Driver.tgsd(X, self.psi, self.phi, self.mask, self.iterations, self.K, self.lambda1, self.lambda2,
                                self.lambda3,
                                self.rho1,
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
        return -np.linalg.norm(residual, ord='fro') / np.linalg.norm(X, ord='fro')

    def get_best_score(self):
        """
        Returns:
            Best score of grid search
        """
        return self.best_score

    def get_best_params(self):
        """
        Returns:
            Best parameter combination of grid search
        """
        return self.best_params

    def run_grid_search(self):
        """
        Uses the parameter grid to determine the best combination of parameters and best score
        """
        param_grid = self.param_grid_mask if len(self.mask) > 0 else self.param_grid_no_mask
        grid_search = GridSearchCV(self, param_grid)
        # Fit parameters to custom encoder
        grid_search.fit(self.X)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        print("Best parameters:", grid_search.best_params_)
        print("Best score (negative Frobenius norm):", grid_search.best_score_)

