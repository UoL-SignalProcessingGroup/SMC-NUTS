from time import time

import autograd.numpy as np
from tqdm import tqdm

from smcnuts.tempering.adaptive_tempering import AdaptiveTempering
from .utils.CheckAttributes import *
from samples import Samples

class SMCSampler():
    """Hamiltonian Monte Carlo SMC Sampler

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

    def __init__(
        self,
        K: int,
        N: int,
        target,
        forward_kernel,
        sample_proposal,
        lkernel="asymptotic",
        tempering=None,
        verbose: bool = False,
        rng = np.random.default_rng(),
    ):
        self.K = K  # Number of iterations
        self.N = N  # Number of particles
        self.target = target  # Target distribution
        self.sample_proposal = sample_proposal  # Initial sample proposal distribution
        self.tempering = tempering  # Tempering scheme
        
        self.samples = Samples(self.N, self.target.dim, self.sample_proposal, self.target, forward_kernel, lkernel, rng)


        self.verbose = verbose  # Show stdout
        self.rng = rng  # Random number generator        

        # This needs to go, better to use templating
        Check_Fwd_Proposal(self.forward_kernel) # Run checks to make sure proposal has attributes to run
        Check_Asym_Has_AccRej(self) # Force asymptoptic L-kernel utilises Accept=reject
        Set_MeanVar_Arrays(self) # Set size of arrays of mean and variance estimates.

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


    def update_sampler(self, k, mean_estimate, variance_estimate, ess):
        "Update the sampler for evaluation purposes."
        self.phi[k] = self.samples.phi_new
        self.log_likelihood[k] = self.samples.log_likelihood
        self.mean_estimate[k] = mean_estimate
        self.variance_estimate[k] = variance_estimate
        self.ess = ess
        

    def sample(self, save_samples=False, show_progress=True):
        """
        Sample from the target distribution using an SMC sampler.
        """

        start_time = time()

        # Main sampling loop
        for k in tqdm(range(self.K), desc=f"NUTS Sampling", disable=not show_progress):
            """
            Prototype code

            self.samples.normalise_weights()
            mean_estimate, variance_estimate = self.estimate(self.samples.x, self.samples.wn)
            self.samples.calculate_ess()
            self.samples.resample()
            self.samples.importance_sampling()
            self.update_sampler()
            """

           
            # Calculate the effective sample size and resample if necessary
            ess = self.calculate_ess(wn)
           

            # Propogate particles through the forward kernel
            r = self.forward_kernel.momentum_proposal.rvs(self.N)

            grad_x = self.target.logpdfgrad(x, phi=phi_new)
            x_new, r_new= self.forward_kernel.rvs(x, r, grad_x, phi=phi_new)

            # Calculate number of accepted particles
            self.acceptance_rate[k] = (
                np.sum(np.all(x_new != x, axis=1)) / self.N
            )

            # Calculate the new temperature
            if self.tempering:
                phi_old = phi_new
                if isinstance(self.tempering, AdaptiveTempering):
                    p_logpdf_x_new_phi_old = self.target.logpdf(x_new, phi=phi_new)
                    args = [x_new, p_logpdf_x_new_phi_old, phi_new]
                phi_new = self.tempering.calculate_phi(args)
                if self.verbose:
                    print(f"Temperature at iteration {k}: {phi_new}")

            # Calculate the new weights
            if self.lkernel == "asymptotic":
                if self.tempering:
                    # Evaluate the tempered target distribution
                    p_logpdf_x_phi_old = self.target.logpdf(x, phi=phi_old)
                    p_logpdf_x_phi_new = self.target.logpdf(x, phi=phi_new)

                    logw_new = logw + p_logpdf_x_phi_new - p_logpdf_x_phi_old
                else:
                    logw_new = logw
            else:
                # Evaluate the target distribution, l kernel and forward kernel
                p_logpdf_x = self.target.logpdf(x)
                p_logpdf_xnew = self.target.logpdf(x_new)

                lkernel_logpdf = self.lkernel.calculate_L(r_new, x_new)
                q_logpdf = self.forward_kernel.logpdf(r)

                logw_new = (
                    logw + p_logpdf_xnew - p_logpdf_x + lkernel_logpdf - q_logpdf
                )

            # Update x and logw
            x = x_new.copy()
            logw = logw_new.copy()


        # Normalise importance weights and calculate the log likelihood
        wn, self.log_likelihood[self.K] = self.normalise_weights(logw)

        # Estimate the mean and variance of the target distribution
        self.mean_estimate[self.K], self.variance_estimate[self.K] = self.estimate(x, wn)

        # Calculate the effective sample size and resample if necessary
        self.ess[self.K] = self.calculate_ess(wn)

        self.phi[self.K] = phi_new

        self.run_time = time() - start_time
