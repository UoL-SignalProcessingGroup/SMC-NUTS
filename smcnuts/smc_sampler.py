from time import time

import autograd.numpy as np
from tqdm import tqdm
from smcnuts.tempering.adaptive_tempering import AdaptiveTempering
from .utils.CheckAttributes import *
from samples import Samples

class SMCSampler():
    """Hamiltonian Monte Carlo (NUTS) SMC Sampler

    An SMC sampler that uses Hamiltonian Monte Carlo (HMC) methods to sample
    from the target distribution of interest.

    Attributes:
        K: Number of iterations.
        N: Number of particles.
        target: Target distribution of interest.
        forward_kernel: Forward kernel used to propagate samples.
        sample_proposal: Distribution to draw the initial samples from (q0).
        lkernel: Approximation method for the optimum L-kernel.
    """
    # CHANGE THE ORDER!
    def __init__(
        self,
        K: int,
        N: int,
        target,
        forward_kernel,
        sample_proposal,
        lkernel,
        tempering=False,
        verbose: bool = False,
        rng = np.random.default_rng(),
    ):
        self.K = K  # Number of iterations
        self.N = N  # Number of particles
        self.target = target  # Target distribution
        
        self.samples = Samples(self.N, self.target.dim, sample_proposal, self.target, forward_kernel, lkernel, rng) 

        # This needs to go, better to use templating
        Check_Fwd_Proposal(self.forward_kernel) # Run checks to make sure proposal has attributes to run
        Set_MeanVar_Arrays(self) # Set size of arrays of mean and variance estimates.

        # Set up arrays to be output when the sampler has finished
        self.resampled = [False] * (self.K + 1)
        self.ess = np.zeros(self.K + 1)
        self.log_likelihood = np.zeros(self.K + 1)
        self.phi = np.zeros(self.K + 1)
        self.acceptance_rate = np.zeros(self.K)
        self.run_time = None


    def estimate(self, x, wn):
        """
        Description:
            Importance sampling estimate of the mean and variance of the
            target distribution.

        Args:
            x: Particle positions.
            wn: Normalised importance weights.

        Returns:
            mean_estimate: Estimated mean of the target distribution.
            variance_estimate: Estimated variance of the target distribution.
        """

        if hasattr(self.target, "constrained_dim"):
            _x = self.target.constrain(x)
        else:
            _x = x.copy()

        mean = wn.T @ _x
        x_shift = _x - mean
        var = wn.T @ np.square(x_shift)

        return mean, var


    def update_sampler(self, k, mean_estimate, variance_estimate):
        "Update the sampler for evaluation purposes."
        self.phi[k] = self.samples.phi_new
        self.log_likelihood[k] = self.samples.log_likelihood
        self.mean_estimate[k] = mean_estimate
        self.variance_estimate[k] = variance_estimate
        self.ess[k] = self.samples.ess
        self.acceptance_rate[k] = (np.sum(np.all(self.samples.x_new != self.samples.x, axis=1)) / self.N) # Calculate number of accepted particles        

    def sample(self, save_samples=False, show_progress=True):
        """
        Sample from the target distribution using an SMC sampler.
        """

        start_time = time()

        # Main sampling loop
        for k in tqdm(range(self.K), desc=f"NUTS Sampling", disable=not show_progress):
            # Normalise the weights
            self.samples.normalise_weights()
            
            # Form estimates
            mean_estimate, variance_estimate = self.estimate(self.samples.x, self.samples.wn)
            
            # Calculate the Effective sample size
            self.samples.calculate_ess()
            
            # Resample if necessary
            self.samples.resample_required()
            
            # Propose new samples
            self.samples.propose_samples()

            # Temper distribution
            self.samples.update_temperature()

            # Reweight samples
            self.samples.reweight()
            
            # Update sampler properties for current iteration
            self.update_sampler(k, mean_estimate, variance_estimate)
            

        # Calculate the final params based on the final proposal step  
        self.samples.normalise_weights()
        mean_estimate, variance_estimate = self.estimate(self.samples.x, self.samples.wn)
        self.samples.calculate_ess()
        # Update sampler properties for the final proposal step
        self.update_sampler(self.K, mean_estimate, variance_estimate)


        self.run_time = time() - start_time
