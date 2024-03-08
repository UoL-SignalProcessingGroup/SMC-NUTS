import numpy as np
from scipy.special import logsumexp
from smcnuts.tempering.adaptive_tempering import AdaptiveTempering

class Samples:
    def __init__(self, N, D, sample_proposal, target, forward_kernel, lkernel, rng) -> None:

        """
        Samples is an object that contains the set of SMC samples and their properties.

        params:
        N: The number of samples
        D: The number of dimensions the samples move in
        x: The location of samples in the target space
        x_new: The location of samples in the target space after a proposal step 
        ess: The number of effective samples
        grad_x: The initial gradient of the samples before a proposal step (needs to be removed)
        r: The momentum at the start of a proposal
        r_new: The momentum at the end of a proposal
        logw: sample weights in log space
        logw_new: sample weights in log space after a proposal
        phi: temperature

        """
        self.N = N
        self.D = D
        self.rng = rng
        self.x_new = np.zeros([self.N, self.D])
        self.r= np.zeros([self.N, self.D])
        self.r_new= np.zeros([self.N, self.D])
        self.grad_x = np.zeros([self.N, self.D])
        self.target=target
        self.ess=0
        self.tempering = True
        self.logw = np.zeros(self.N)
        self.logw_new = np.zeros(self.N)

        self.initialise_samples(sample_proposal, target)

        
    def initialise_samples(self, sample_proposal, target):
        self.x = sample_proposal.rvs(self.N)
        self.phi_old, self.phi_new = (0.0, 0.0) if self.tempering else (1.0, 1.0) 
        if self.tempering:
            p_logpdf_x_phi = self.target.logpdf(self.x, phi=self.phi_old)
            self.phi_new = self.tempering.calculate_phi([self.x, p_logpdf_x_phi,self.phi_old])
        p_logpdf_x = self.target.logpdf(self.x, phi=self.phi_new)
        q0_logpdf_x = sample_proposal.logpdf(self.x)
        self.logw = p_logpdf_x - q0_logpdf_x

    
    
    def normalise_weights(self):
        """
        Normalises the sample weights in log scale
        """

        index = ~np.isneginf(self.logw)

        log_likelihood = logsumexp(self.logw[index])

        # Normalise the weights
        wn = np.zeros_like(self.logw)
        wn[index] = np.exp(self.logw[index] - log_likelihood)

        self.wn =wn
        self.log_likelihood=log_likelihood


    def calculate_ess(self):
        """
        Calculate the effective sample size using the normalised
        sample weights.
        """
        self.ess = 1 / np.sum(np.square(self.wn))

    def resample_required(self):
        if(self.ess < self.N / 2):
            self.x, self.logw = self.resample(x, wn, self.log_likelihood[k])
            return True
        return False

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

    def importance_sampling(self):
        pass