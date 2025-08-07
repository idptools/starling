import sys

import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.special import logsumexp

# Currently only single observable is supported
# If multiple observables are needed, the data will need to be z-scored (or min-maxed), so that larger
# magnitude observables do not dominate the optimization and theta can be set globally


class BME:
    def __init__(
        self, experimental_data, calculated_data, theta=0.5, max_iterations=50000
    ):
        self.experimental_data = experimental_data
        self.calculated_data = calculated_data
        self.theta = theta
        self.max_iterations = max_iterations
        self.weights = (
            np.ones(self.calculated_data.shape[0]) / self.calculated_data.shape[0]
        )

        self.lambdas = np.zeros(self.experimental_data.shape[0], dtype=np.longdouble)
        print("Lagrange multipliers initialized from zero\n")

        self.bounds = []
        for j in range(self.experimental_data.shape[0]):
            if self.experimental_data[j, 2] == 0:
                self.bounds.append([None, None])
            elif self.experimental_data[j, 2] == -1:
                self.bounds.append([None, 0.0])
            else:
                self.bounds.append([0.0, None])

    def srel(sefl, w0, w1):
        idxs = np.where(w1 > 1.0e-50)
        return np.sum(w1[idxs] * np.log(w1[idxs] / w0[idxs]))

    def get_chi(self):
        calc_avg = np.sum(self.calculated_data[:, 0] * self.weights, axis=0)

        exp_mean = self.experimental_data[0, 0]
        exp_std = self.experimental_data[0, 1]
        constraint = self.experimental_data[0, 2]

        diff = np.sum(calc_avg - exp_mean)  # Ensures scalar

        # Apply directional constraint logic
        if (
            (diff < 0 and constraint < 0)
            or (diff > 0 and constraint > 0)
            or (constraint == 0)
        ):
            ff = 1
        else:
            ff = 0

        diff *= ff

        return (diff / exp_std) ** 2

    def maxent(self, lambdas):
        # weights
        unnormalized_weights = (
            -np.sum(lambdas * self.calculated_data, axis=1)
            - self.tmax
            + np.log(self.weights)
        )

        normalization_constant = logsumexp(unnormalized_weights)
        normalized_weights = np.exp(unnormalized_weights - normalization_constant)

        avg = np.sum(normalized_weights[:, np.newaxis] * self.calculated_data, axis=0)

        # gaussian integral
        theta_sigma2 = np.sum((lambdas**2 * self.experimental_data[:, 1] ** 2))
        eps2 = self.theta / 2 * theta_sigma2

        # experimental value
        sum1 = np.dot(lambdas, self.experimental_data[:, 0])

        fun = sum1 + eps2 + normalization_constant

        # gradient
        jac = self.experimental_data[:, 0] + lambdas * theta_sigma2 - avg

        # divide by theta to avoid numerical problems
        return fun / self.theta, jac / self.theta

    def fit(self):
        self.tmax = np.log((sys.float_info.max) / 5.0)

        chi2_before = self.get_chi()

        print("CHI2 before optimization: %8.4f \n" % (chi2_before))
        # self.log.flush()
        mini_method = "L-BFGS-B"

        result = minimize(
            self.maxent,
            self.lambdas,
            options={"maxiter": 50000, "disp": False},
            method=mini_method,
            jac=True,
            bounds=self.bounds,
        )
        if result.success:
            print(
                "Minimization using %s successful (iterations:%d)\n"
                % (mini_method, result.nit)
            )
            arg = (
                -np.sum(result.x[np.newaxis, :] * self.calculated_data, axis=1)
                - self.tmax
            )
            w_opt = self.weights * np.exp(arg)
            w_opt /= np.sum(w_opt)
            self.lambdas = np.copy(result.x)
            self.w_opt = np.copy(w_opt)
            self.weights = w_opt
            self.niter = result.nit
            chi2_after = self.get_chi()
            phi = np.exp(-self.srel(self.weights, w_opt))

            print("CHI2 after optimization: %8.4f \n" % (chi2_after))
            print("Fraction of effective frames: %8.4f \n" % (phi))

            return chi2_before, chi2_after, phi

        else:
            print("Minimization using %s failed\n" % (mini_method))
            print("Message: %s\n" % (result.message))
            self.niter = -1
            return np.nan, np.nan, np.nan

    def predict(self, new_data):
        """
        Predict the observable for new data using the optimized weights.
        :param new_data: New data points to predict the observable for.
        :return: Predicted observable values.
        """
        if self.w_opt is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return (self.w_opt * new_data).sum()
