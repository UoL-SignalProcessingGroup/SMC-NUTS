import autograd.numpy as np


class ESSRecycling:
    """Effective Sample Size Recycling

    Recycle samples generated at all iterations of the SMC sampler using
    the effective sample size (ESS) recycling scheme. See [1] for details.

    [1] Nguyen, T., Septier, F., Peters, G. and Delignon, Y. (2014). Improving
    SMC sampler estimate by recycling all past simulated samples.
    """

    def __init__(self, K: int, target):
        self.K = K + 1
        self.target = target

    def constant(self, wn):
        """
        Description:
            Calculate the ESS recycling constant.

        Args:
            wn: Normalised importance weights.

        Returns:
            ess_constant: Constant used in the ESS recycling scheme.
        """

        return 1 / np.sum(np.square(wn))  # type: ignore

    def recycle_mean(self, mean_estimates, constants):
        """
        Description:
            Recycle the mean estimates.

        Args:
            mean_estimates: Mean estimates.
            constants: Constants used in the ESS recycling scheme.

        Returns:
            recycled_mean: Recycled mean estimates.
        """
        if hasattr(self.target, "constrained_dim"):
            recycled_estimates = np.zeros([self.K, self.target.constrained_dim])
        else:
            recycled_estimates = np.zeros([self.K, self.target.dim])
        for k in range(self.K):
            ld = np.zeros((k + 1,))
            _sum = np.sum(constants[: k + 1])
            for i in range(k + 1):
                ld[i] = constants[i] / _sum
            recycled_estimates[k] = ld @ mean_estimates[: k + 1, :]

        return recycled_estimates  # type: ignore

    def recycle_variance(self, var_estimates, mean_estimates, recycled_mean, constants):
        """
        Description:
            Recycle the variance estimates.

        Args:
            var_estimates: Variance estimates.
            mean_estimates: Mean estimates.
            recycled_mean: Recycled mean estimates.
            constants: Constants used in the ESS recycling scheme.

        Returns:
            recycled_var: Recycled variance estimates.
        """

        if hasattr(self.target, "constrained_dim"):
            recycled_estimates = np.zeros([self.K, self.target.constrained_dim])
        else:
            recycled_estimates = np.zeros([self.K, self.target.dim])
        for k in range(self.K):
            ld = np.zeros((k + 1,))
            _sum = np.sum(constants[: k + 1])
            if hasattr(self.target, "constrained_dim"):
                correction = np.zeros([self.K, self.target.constrained_dim])
            else:
                correction = np.zeros([self.K, self.target.dim])
            for i in range(0, k + 1):
                ld[i] = constants[i] / _sum
                correction[i] = np.square(recycled_mean[k] - mean_estimates[i])
            recycled_estimates[k] = ld @ (
                var_estimates[: k + 1, :] + correction[: k + 1, :]
            )

        return recycled_estimates
