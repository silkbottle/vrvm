import numpy as np


from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel
)


class RVR(BaseEstimator):

    def __init__(
        self,
        kernel='rbf',
        degree=3,
        gamma=0.1,
        n_iter=3000,
        tol=1e-3,
        n_tries=3,
        threshold_alpha=1e9,
        bias_used=True,
        verbose=False
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.n_iter = n_iter
        self.tol = tol
        self.n_tries=n_tries,
        self.threshold_alpha = threshold_alpha
        self.bias_used = bias_used
        self.verbose = verbose

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'n_tries': self.n_tries,
            'threshold_alpha': self.threshold_alpha,
            'bias_used': self.bias_used,
            'verbose': self.verbose
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _apply_kernel(self, x, y):
        """Apply the selected kernel function to the data."""
        if self.kernel == 'linear':
            phi = linear_kernel(x, y)
        elif self.kernel == 'rbf':
            phi = rbf_kernel(x, y, self.gamma)
        elif self.kernel == 'poly':
            phi = polynomial_kernel(x, y, self.degree)
        elif callable(self.kernel):
            phi = self.kernel(x, y)
            if len(phi.shape) != 2:
                raise ValueError(
                    "Custom kernel function did not return 2D matrix"
                )
            if phi.shape[0] != x.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows"
                    " equal to number of data points."""
                )
        else:
            raise ValueError("Kernel selection is invalid.")

        if self.bias_used:
            phi = np.hstack((np.ones((phi.shape[0], 1)), phi))

        return phi

    def _prune(self):

        self.tries[self.alpha_ > self.threshold_alpha] += 1
        keep = self.tries < self.n_tries
        self.tries = self.tries[keep]
        self.relevance_ = self.relevance_[keep]
        self.alpha_ = self.alpha_[keep]
        self.alpha_[self.alpha_ > self.threshold_alpha] = self.threshold_alpha
        self.phi = self.phi[:, keep]

    def fit(self, X, y):

        self.X = X
        self.y = y

        self.phi = self._apply_kernel(X, X)

        n_samples, n_features = self.phi.shape

        self.relevance_ = np.arange(n_features)
        self.tries = np.zeros(n_features)
        self.alpha_ = np.random.rand(n_features)
        self.beta_ = np.random.rand(1)

        self.mu_ = np.zeros(n_features)

        for i in range(self.n_iter):
            
            i_s = np.diag(self.alpha_) + self.beta_ * np.dot(self.phi.T, self.phi)
            self.sigma_ = np.linalg.inv(i_s)
            self.mu_ = self.beta_ * np.dot(self.sigma_, np.dot(self.phi.T, self.y))

            gamma = 1 - self.alpha_*np.diag(self.sigma_)
            self.alpha_ = gamma/(self.mu_ ** 2)

            self.beta_ = (n_samples - np.sum(gamma))/(np.sum((y - np.dot(self.phi, self.mu_)) ** 2))

            self._prune()

        return self

    def evidence(self):
        S = 1. / self.beta_ * np.eye(self.phi.shape[0]) + self.phi.dot(self.phi.T/self.alpha_.reshape(-1, 1))
        S_inv = np.linalg.inv(S)
        exact_evidence = -np.log(np.diag(np.linalg.cholesky(S))).sum() - 0.5 * self.y.T.dot(S_inv).dot(self.y) - 0.5 * self.phi.shape[0]*np.log(2 * np.pi)
        upper_bound = - 0.5 * self.y.T.dot(S_inv).dot(self.y) + 0.5 * self.phi.shape[0] * (np.log(self.beta_) - np.log(2 * np.pi))
        return exact_evidence, upper_bound

    def predict(self, x):

        if self.bias_used:
            if self.relevance_[0] == 0:
                phi = self._apply_kernel(self.X[self.relevance_[1:] - 1], x)
            else:
                phi = self._apply_kernel(self.X[self.relevance_ - 1], x)[:, 1:]
        else:
            phi = self._apply_kernel(self.X[self.relevance_], x)

        return phi.T.dot(self.mu_)
