import numpy as np
import scipy.optimize as scopt
from sklearn.linear_model.base import BaseEstimator
from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel
)

class RVM(BaseEstimator):

	def __init__(
        self,
        kernel='linear',
        degree=3,
        gamma=None,
        n_iter=3000,
        tol=1e-3,
        alpha=1e-6,
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
		self.alpha = alpha
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
		    'alpha': self.alpha,
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


	def _apply_kernel(self, X, y=None):
		"""Apply the selected kernel function to the data."""
		if self.kernel == 'linear':
		    phi = linear_kernel(X, y)
		elif self.kernel == 'rbf':
		    phi = rbf_kernel(X, y, gamma=self.gamma)
		elif self.kernel == 'poly':
		    phi = polynomial_kernel(X, y, degree=self.degree)
		elif callable(self.kernel):
		    phi = self.kernel(X, y)
		    if len(phi.shape) != 2:
		        raise ValueError(
		            "Custom kernel function did not return 2D matrix"
		        )
		    if phi.shape[0] != X.shape[0]:
		        raise ValueError(
		            "Custom kernel function did not return matrix with rows"
		            " equal to number of data points."""
		        )
		else:
		    raise ValueError("Kernel selection is invalid.")

		return phi

	def _upper_bound(self, X, y):

		Y = X.copy()
		for i in xrange(X.shape[0]):
			Y[i] *= y[i]

		def bound(l):
			w = (l * y).dot(X)
			return ((1 - l) * np.log(1 - l) + l * np.log(l)).sum() + 0.5 * w.dot(self.alpha * w)

		def gradient(l):
			return np.log(l) - np.log(1 - l) + Y.dot(self.alpha * (l * y).dot(X))

		def part(l):
			s = np.zeros(X.shape[1])
			for i in xrange(X.shape[0]):
				s += l[i] * y[i] * X[i]
			s *= self.alpha
			print s - self.alpha * (l * y).dot(X)

		# def hessian(l):
		# 	return np.diag(1./ l / (1 - l)) + Y.dot(np.diag(self.alpha).dot(Y.T))
		eps = 1e-6
		options = {'maxiter': 1000 * X.shape[0], 'xtol': eps}
		l_opt = scopt.minimize(bound, 0.5 * np.random.rand(X.shape[1]), method='TNC', jac=gradient, bounds=[(eps, 1-eps)] * X.shape[0], options=options)['x']
		return bound(l_opt), l_opt

	def _fit(self, X, y):

		def f(x):
			return np.tanh(x) / x / 4

		def sigm(x):
			return 1. / (1 + np.exp(-x))

		def lower_bound(ksi, S, S_inv, mu, alpha):
			ret = 0.5 * np.log(alpha).sum() + np.log(sigm(ksi)).sum()
			ret += (f(ksi) * ksi ** 2).sum()
			ret -= 0.5 * ksi.sum()
			ret += 0.5 * mu.T.dot(S_inv).dot(mu)
			ret += 0.5 * np.log(np.linalg.det(S))
			return ret

		n_objects = X.shape[0]
		n_features = X.shape[1]

		alpha = np.random.rand(n_features)
		ksi = np.random.rand(n_objects) * 100

		for it in xrange(100):
			S_inv = np.diag(alpha) + X.T.dot(np.diag(f(ksi)).dot(X))
			S = np.linalg.inv(S_inv)
			m = 0.5 * S.dot(X.T.dot(y))
			sigma = np.outer(m, m) + S
			# alpha = 1. / np.diagonal(sigma) 
			for i in xrange(n_objects):
				ksi[i] = np.sqrt(X[i].dot(sigma).dot(X[i].T))
			# print alpha.min(), alpha.max()
			# print 'ksi', ksi.min(), ksi.max()
			print lower_bound(ksi, S, S_inv, m, alpha)

		self.alpha = alpha

	def fit(self, X, y):
		self.X = X
		phi = self._apply_kernel(X)
		n_objects = phi.shape[0]
		n_features = phi.shape[1]
		self._fit(phi, y)
		# _, l = self._upper_bound(phi, y)
		return 1

	def predict(self, x):
		phi = self._apply_kernel(self.X, x)
		

