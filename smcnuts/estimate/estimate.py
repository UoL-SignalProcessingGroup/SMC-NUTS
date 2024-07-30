import numpy as np


class Estimate():
    """Estimate class

    Description: Calculates values of interest, i.e. expection values and variance, of the target distribution from a set of weighted samples

    Attributes:
        K: Number of iterations.
        N: Number of particles.
        target: Target distribution of interest.
        sample_proposal: Distribution to draw the initial samples from (q0).
        lkernel: Approximation method for the optimum L-kernel.
    """

    def __init__(
        self,
        target
    ):

        self.target = target

        # Check if target has constrained parameters
        if hasattr(self.target, "constrained_dim"):
            self._sample_space = self._constrained_target
        else:
            self._sample_space = self._unconstrained_target

    def _constrained_target(self, x):
        # Find values of samples in constrained space
        return self.target.constrain(x)

    def _unconstrained_target(self, x):
        # Return a copy of the samples if samples have no constrained parameters
        return x.copy()

    def return_estimate(self, x, wn):
        """
        Description:
            Importance sampling estimate of the mean and variance of the
            target distribution in constrained space (if required).

        Args:
            x: Particle positions.
            wn: Normalised importance weights.

        Returns:
            mean_estimate: Estimated mean of the target distribution.
            variance_estimate: Estimated variance of the target distribution.
        """

        _x = self._sample_space(x)

        mean, var = self._estimate(_x, wn)

        return mean, var

    def return_estimate_unconstrained(self, x, wn):
        """
        Description:
            Importance sampling estimate of the mean and variance of the
            target distribution in unconstrained space (i.e. the space the samples are moving in).
            Primarily used for obtaing the covariance of the space the samples are moving in

        Args:
            x: Particle positions.
            wn: Normalised importance weights.

        Returns:
            mean_estimate: Estimated mean of the target distribution.
            variance_estimate: Estimated variance of the target distribution.
        """

        mean, var = self._estimate(x, wn)

        return mean, var

    def _estimate(self, x, wn):
        """
        Description:
            Returns mean and variance estimate from normalised weights
        Args:
            x: Particle positions.
            wn: Normalised importance weights.

        Returns:
            mean_estimate: Estimated mean of the target distribution.
            variance_estimate: Estimated variance of the target distribution.
        """
        mean = wn.T @ x
        x_shift = x - mean
        var = wn.T @ np.square(x_shift)

        return mean, var
