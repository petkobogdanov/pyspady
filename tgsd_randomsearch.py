import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import tgsd_home
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

TGSD_Driver = tgsd_home.TGSD_Home


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config_path, K=7, iterations=100, lambda1=0.1, lambda2=0.1, lambda3=1, rho1=0.01, rho2=0.01):
        self.config_path = config_path
        self.tgsd_driver = tgsd_home.TGSD_Home(config_path=self.config_path)
        self.X = self.tgsd_driver.X
        self.Psi_D = self.tgsd_driver.Psi_D
        self.Phi_D = self.tgsd_driver.Phi_D
        self.mask = self.tgsd_driver.mask
        self.K = K
        self.iterations = iterations
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.rho1 = rho1
        self.rho2 = rho2
        self.best_params = None
        self.best_score = None
        self.param_distributions_mask = {
            'K': sp_randint(1, 11),
            'lambda1': uniform(0.001, 1),
            'lambda2': uniform(0.001, 1),
            'lambda3': uniform(0.1, 10)
        }
        self.param_distributions_no_mask = {
            'K': sp_randint(1, 11),
            'lambda1': uniform(0.001, 1),
            'lambda2': uniform(0.001, 1),
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
        Y, W = TGSD_Driver.tgsd(X, self.Psi_D, self.Phi_D, self.mask, self.iterations, self.K, self.lambda1, self.lambda2,
                                self.lambda3, self.rho1,
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
        residual = X - (self.Psi_D @ self.Y_ @ self.W_ @ self.Phi_D)
        residual_percentage = -np.linalg.norm(residual, ord='fro') / np.linalg.norm(X, ord='fro')
        return residual_percentage

    def run_random_search(self):
        """
        Uses the parameter distributions to determine the best combination of parameters and best score
        """
        # n_iters denotes how many times a permutation is chosen across cv #folds
        param_distributions = self.param_distributions_mask if len(self.mask) > 0 else self.param_distributions_no_mask
        random_search = RandomizedSearchCV(estimator=self, param_distributions=param_distributions, n_iter=100, cv=5, random_state=42)
        # Fit parameters to custom encoder
        random_search.fit(self.X)
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        print("Best parameters:", random_search.best_params_)
        print("Best score (negative Frobenius norm):", random_search.best_score_)
