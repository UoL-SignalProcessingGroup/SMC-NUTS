import autograd.numpy as np
from scipy.optimize import bisect
from scipy.special import logsumexp


class AdaptiveTempering():
    def __init__(self, N, target, alpha=0.5):
        self.N = N  # Number of particles
        self.target = target  # Target
        self.alpha = alpha  # ESS threshold

    def calculate_phi(self, args):
        """
        Description
        -----------
        Calculate the new temperature by maximising the ESS
        of the current set of samples.

        Parameters
        ----------
        args: Arguments required for this tempering scheme
              (old_temperature, p_logpdf_x, alpha)

        Returns
        -------
        opt_lambda: value of lambda that maximises the ESS
        """

        x_new, p_logpdf_x_new_phi_old, old_phi = args

        #precalculate the log prior and likeloihood for efficiency
        logpri= self.target.logpdf(x_new, phi=0.0)
        loglik= self.target.logpdf(x_new, phi=1.0) - logpri

        def _ess(new_phi):
            # Calculate logw at the new temperature
            logw = new_phi*loglik + logpri - p_logpdf_x_new_phi_old
            
            # Normalise the weights
            index = ~np.isneginf(logw)

            log_likelihood = logsumexp(logw[index])

            # Normalise the weights
            wn = np.exp(logw[index] - log_likelihood)

            # Calculate the ESS
            ESS = 1 / np.sum(np.square(wn))
            
            return ESS - self.N * self.alpha 

        if _ess(1.0) >= 0:
            return 1.0

        opt_lambda = bisect(_ess, old_phi, 1.0)

        return opt_lambda
