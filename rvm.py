import numpy as np
import scipy.optimize as scopt
from scipy.misc import logsumexp
from sklearn.linear_model.base import BaseEstimator
from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel
)

def f(x):
    return (sigma(x) - 0.5) / (2. * x + 1e-8)

def sigma(x):
    return 1. / (1. + np.exp(-x))

class RVM(BaseEstimator):

    def __init__(
        self,
        kernel='linear',
        degree=3,
        gamma=None,
        n_iter=3000,
        n_tries=3,
        tol=1e-3,
        threshold_alpha=1e9,
        bias_used=True,
        fast_opt=True,
        verbose=False
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.n_iter = n_iter
        self.n_tries = 3
        self.tol = tol
        self.threshold_alpha = threshold_alpha
        self.bias_used = bias_used
        self.fast_opt = fast_opt
        self.verbose = verbose

        self.upper_bound = np.zeros(n_iter)
        self.lower_bound = np.zeros(n_iter)

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'threshold_alpha': self.threshold_alpha,
            'bias_used': self.bias_used,
            'fast_opt': self.fast_opt,
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
        phi = phi.T
        if self.bias_used:
            phi = np.hstack((np.ones((phi.shape[0], 1)), phi))

        return phi

    def _wmp(self):

        X = np.diag(self.y).dot(self.phi_)

        def cost(w):
            return np.log(1 + np.exp(-X.dot(w))).sum() + 0.5 * (self.alpha_ * w * w).sum()

        def gradient(w):
            ret = self.alpha_ * w
            z = X.dot(w)
            num = np.zeros(z.shape)
            num[z > 0] = np.exp(-z[z > 0]) / (1. + np.exp(-z[z > 0]))
            num[z <= 0] = 1. / (1. + np.exp(z[z <= 0]))
            ret -= (np.multiply(num.reshape(-1, 1), X)).sum(axis=0)
            return ret

        result = scopt.minimize(cost, self.mu_, method='BFGS', jac=gradient, tol=1e-3, options={'maxiter': 100})
        wmp = result['x']
        z = X.dot(wmp)
        self.l = np.zeros(z.shape)
        self.l[z > 0] = np.exp(-z[z > 0]) / (1. + np.exp(-z[z > 0]))
        self.l[z <= 0] = 1. / (1. + np.exp(z[z <= 0]))
        self.w_ = wmp 
        return -cost(wmp)

    def evidence(self, n_points=10, n_epoch=10):
        sigma_inv = np.linalg.inv(self.sigma_)
        results = np.zeros(n_epoch)
        for epoch in xrange(n_epoch):
            ret = 0.5 * np.log(self.alpha_).sum()
            ret += np.log(np.diag(np.linalg.cholesky(self.sigma_))).sum()
            samples = np.random.multivariate_normal(self.mu_, self.sigma_, n_points)
            X = self.y.reshape(-1, 1) * self.phi_
            logs = np.zeros(n_points)
            for i in xrange(n_points):
                logs[i] = -.5 * samples[i].T.dot(self.alpha_ * samples[i])
                logs[i] -= np.log(1. + np.exp(-X.dot(samples[i]))).sum()
                logs[i] += .5 * (samples[i] - self.mu_).T.dot(sigma_inv).dot(samples[i] - self.mu_)
                # print logs[i]
            ret += logsumexp(logs)
            ret -= np.log(1. * n_points)
            results[epoch] = ret

        return results.mean(), results.std()

    def upper_bound_mc(self, n_points=10, n_epoch=10):
        sigma_inv = np.linalg.inv(self.sigma_)
        results = np.zeros(n_epoch)
        for epoch in xrange(n_epoch):
            ret = 0.5 * np.log(self.alpha_).sum()
            ret += np.log(np.diag(np.linalg.cholesky(self.sigma_))).sum()
            samples = np.random.multivariate_normal(self.mu_, self.sigma_, n_points)
            X = self.y.reshape(-1, 1) * self.phi_
            logs = np.zeros(n_points)
            for i in xrange(n_points):
                logs[i] = -.5 * samples[i].T.dot(self.alpha_ * samples[i])
                logs[i] -= np.log(1. + np.exp(-X.dot(samples[i]))).sum()
                logs[i] += .5 * (samples[i] - self.mu_).T.dot(sigma_inv).dot(samples[i] - self.mu_)
                # print logs[i]
            ret += logsumexp(2. * logs)
            ret -= np.log(1. * n_points)
            results[epoch] = ret / 2.

        return results.mean(), results.std()

    # def _upper_bound(self):

    #     Y = self.phi_.copy()
    #     for i in xrange(self.phi_.shape[0]):
    #         Y[i] *= self.y[i]

    #     def bound(l):
    #         w = (l * self.y).dot(self.phi_)
    #         return ((1 - l) * np.log(1 - l) + l * np.log(l)).sum() + 0.5 * w.dot(w/self.alpha_)

    #     def gradient(l):
    #         return np.log(l) - np.log(1 - l) + Y.dot((l * self.y).dot(self.phi_) / self.alpha_)

    #     eps = 1e-8
    #     options = {'maxiter': 1000 * self.phi_.shape[0], 'xtol': eps}
    #     l_opt = scopt.minimize(bound, 0.5 * np.random.rand(self.phi_.shape[0]), method='TNC', jac=gradient, bounds=[(eps, 1-eps)] * self.phi_.shape[0], options=options)['x']

    #     self.l = l_opt

    #     return bound(l_opt), l_opt

    # def _upper_bound_(self):

    #     X = np.diag(self.y).dot(self.phi_)
    #     A = X.dot(np.diag(1./ self.alpha_)).dot(X.T)
    #     x = cvx.Variable(self.phi_.shape[0])
    #     objective = cvx.Minimize(-cvx.sum_entries(cvx.entr(x) + cvx.entr(1 - x)) + 0.5 * cvx.quad_form(x, A))
    #     constraints = [0 < x, x < 1]
    #     prob = cvx.Problem(objective, constraints)
    #     result = prob.solve(solver=cvx.CVXOPT)

    #     return prob.value

    def _prune(self):

        self.tries[self.alpha_ > self.threshold_alpha] += 1
        keep = self.tries < self.n_tries
        self.tries = self.tries[keep]
        self.relevance_ = self.relevance_[keep]
        self.alpha_ = self.alpha_[keep]
        self.alpha_[self.alpha_ > self.threshold_alpha] = self.threshold_alpha
        self.phi_ = self.phi_[:, keep]

    def _fit(self, X, y):

        def lower_bound(ksi, S, S_inv):
            ret = (np.log(sigma(ksi)) + f(ksi) * ksi ** 2 - 0.5*ksi).sum()
            ret += 0.5 * np.log(self.alpha_).sum()
            ret -= np.log(np.diag(np.linalg.cholesky(S_inv))).sum()
            # print ret, self.phi_.T.dot(y).T.dot(S).dot(self.phi_.T.dot(y)) / 8, 0.5 * np.log(np.linalg.det(np.eye(self.phi_.shape[1]) + 2.*np.diag(1./self.alpha_).dot(self.phi_.T).dot(np.diag(f(ksi)).dot(self.phi_))))
            # ret -= 0.5 * np.log(np.linalg.det(np.eye(self.phi_.shape[1]) + 2.*np.diag(1./self.alpha_).dot(self.phi_.T).dot(np.diag(f(ksi)).dot(self.phi_))))
            ret += 0.5 * self.mu_.T.dot(S_inv).dot(self.mu_)
            return ret

        def lower_bound_(ksi, S, S_inv):
            ret = (np.log(sigma(ksi)) + f(ksi) * ksi ** 2 - 0.5*ksi).sum()
            ret += 0.5 * np.log(self.alpha_).sum()
            ret -= np.log(np.diag(np.linalg.cholesky(S_inv))).sum()
            for i in xrange(n_objects):
                ret -= f(ksi[i]) * (self.phi_[i].dot(self.mu_) ** 2 + self.phi_[i].dot(S).T.dot(self.phi_[i]))
            ret += 0.5 * self.mu_.dot(self.phi_.T.dot(y))
            ret -= 0.5 * ((self.mu_ * self.mu_ + np.diagonal(S)) * self.alpha_).sum()
            # print 'strange', 0.5 * np.log(self.alpha_).sum() - 0.5 * ((self.mu_ * self.mu_ + np.diagonal(S)) * self.alpha_).sum()
            ret += self.phi_.shape[1]/2
            return ret

        self.phi_ = self._apply_kernel(X)
        self.y = y
        n_objects = self.phi_.shape[0]
        n_features = self.phi_.shape[1]
        self.relevance_ = np.arange(n_features)
        self.tries = np.zeros(n_features)
        self.alpha_ = np.random.rand(n_features)
        ksi = np.random.rand(n_objects)

        for it in xrange(self.n_iter):
            S_inv = np.diag(self.alpha_) + 2. * self.phi_.T.dot(np.diag(f(ksi)).dot(self.phi_))
            S = np.linalg.inv(S_inv)
            self.mu_ = 0.5 * S.dot(self.phi_.T.dot(y))
            # self.lower_bound[it] = lower_bound_(ksi, S, S_inv)
            # self.upper_bound[it] = self._wmp()
            # print '1', lower_bound_(ksi, S, S_inv)
            if self.fast_opt:
                self.alpha_ = (1. - np.diagonal(S) * self.alpha_) / (self.mu_ * self.mu_)
            else:
                self.alpha_ = 1. / (self.mu_ * self.mu_ + np.diagonal(S)) 
            # print '2', lower_bound_(ksi, S, S_inv)
            for i in xrange(n_objects):
                ksi[i] = np.sqrt(self.phi_[i].dot(self.mu_) ** 2 + self.phi_[i].dot(S).T.dot(self.phi_[i]))
            self._prune()

        S_inv = np.diag(self.alpha_) + 2. * self.phi_.T.dot(np.diag(f(ksi)).dot(self.phi_))
        S = np.linalg.inv(S_inv)
        self.mu_ = 0.5 * S.dot(self.phi_.T.dot(y))
        self.sigma_ = S
        self.lower_bound[-1] = lower_bound_(ksi, S, S_inv)
        self.upper_bound[-1] = self._wmp()
        self.ksi = ksi

    def lb(self, w):
        ret = np.log(self.alpha_).sum() / 2
        ret += (np.log(sigma(self.ksi)) + f(self.ksi) * self.ksi ** 2 - 0.5*self.ksi).sum()
        ret -= (self.alpha_ * w * w).sum() / 2
        ret -= w.T.dot(self.phi_.T.dot(np.diag(f(self.ksi)).dot(self.phi_))).dot(w)
        ret += w.dot(self.phi_.T.dot(self.y)) / 2
        ret -= np.log(2 * np.pi)*self.phi_.shape[1] / 2
        return ret

    def ub(self, w):
        ret = np.log(self.alpha_).sum() / 2
        ret -= (self.alpha_ * w * w).sum() / 2
        ret += ((1 - self.l) * np.log(1 - self.l) + self.l * np.log(self.l)).sum()
        ret += (np.diag(self.l * self.y).dot(self.phi_).dot(w)).sum()
        ret -= np.log(2 * np.pi)*self.phi_.shape[1] / 2
        return ret 

    def f(self, w):
        ret = np.log(self.alpha_).sum() / 2
        ret -= (self.alpha_ * w * w).sum() / 2
        ret -= np.log(1 + np.exp(-np.diag(self.y).dot(self.phi_).dot(w))).sum()
        ret -= np.log(2 * np.pi)*self.phi_.shape[1] / 2
        return ret

    def fit(self, X, y):
        self.X = X
        self._fit(X, y)

    def predict_proba(self, x):
        if self.bias_used:
            if self.relevance_[0] == 0:
                phi = self._apply_kernel(self.X[self.relevance_[1:] - 1], x)
            else:
                phi = self._apply_kernel(self.X[self.relevance_ - 1], x)[:, 1:]
        else:
            phi = self._apply_kernel(self.X[self.relevance_], x)
        return sigma(phi.dot(self.mu_))

    def predict(self, x):
        return 2 * (self.predict_proba(x) > 0.5) - 1
        

