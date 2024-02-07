import numpy as np
from scipy.special import logsumexp

class Samples:
    def __init__(self, N, D, sample_proposal, target) -> None:
        self.N = N
        self.D = D
        self.x_new = np.zeros([self.N, self.D])
        self.r= np.zeros([self.N, self.D])
        self.r_new= np.zeros([self.N, self.D])
        self.grad_x = np.zeros([self.N, self.D])
        
        self.ess=0

        self.logw = np.zeros(self.N)
        self.logw_new = np.zeros(self.N)

        self.initialise_samples(sample_proposal, target)
        
        
    def initialise_samples(self, sample_proposal):
        self.x = sample_proposal.rvs(self.N)
        
        p_logpdf_x = self.target.logpdf(self.x, phi=phi_new)
        q0_logpdf_x = self.sample_proposal.logpdf(self.x)
        self.logw = p_logpdf_x - q0_logpdf_x

    def normalise_weights(self, logw):
        """
        Normalises the sample weights

        Args:
            logw: A list of sample weights on the log scale

        Returns:
            A list of normalised weights

        """

        index = ~np.isneginf(logw)

        log_likelihood = logsumexp(logw[index])

        # Normalise the weights
        wn = np.zeros_like(logw)
        wn[index] = np.exp(logw[index] - log_likelihood)

        return wn, log_likelihood  # type: ignore

    def calculate_ess(self, wn):
        """
        Calculate the effective sample size using the normalised
        sample weights.

        Args:
            wn: A list of normalised sample weights

        Return:
            The effective sample size
        """

        ess = 1 / np.sum(np.square(wn))

        return ess

    def resample(self, x, wn, log_likelihood):
        """
        Resamples samples and their weights from the specified indexes.

        Args:
            x: A list of samples to resample
            wn: A list of normalise sample weights to resample
            indexes: A list of the indexes of samples and weights to resample

        Returns:
            x_new: A list of resampled samples
            logw_new: A list of resampled weights
        """

        # Resample x
        i = np.linspace(0, self.N-1, self.N, dtype=int)
        i_new = self.rng.choice(i, self.N, p=wn)
        x_new = x[i_new]

        # Determine new weights
        # logw_new = np.log(np.ones(self.N_local)) - self.N_local
        logw_new = (np.ones(self.N) * log_likelihood) - np.log(self.N)

        return x_new, logw_new