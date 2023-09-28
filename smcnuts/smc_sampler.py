from time import time

import autograd.numpy as np
from tqdm import tqdm
import warnings
from scipy.special import logsumexp

from smcnuts.recycling.ess import ESSRecycling
from smcnuts.tempering.adaptive_tempering import AdaptiveTempering


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
        recycling: Particle recycling method.
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
        recycling=None,
        verbose: bool = False,
        rng = np.random.default_rng(),
    ):
        self.K = K  # Number of iterations
        self.N = N  # Number of particles
        self.target = target  # Target distribution
        self.forward_kernel = forward_kernel  # Forward kernel distribution
        self.sample_proposal = sample_proposal  # Initial sample proposal distribution
        self.tempering = tempering  # Tempering scheme
        self.lkernel = lkernel  # L-kernel distribution
        self.recycling = recycling  # Recycling scheme
        self.verbose = verbose  # Show stdout
        self.rng = rng  # Random number generator

        if self.lkernel == "asymptotic" and self.forward_kernel.accept_reject == False:
            warnings.warn("Warning: Accept-reject is false and therefore not a valid MCMC kernel. Setting accept-reject to true.")
            self.forward_kernel.accept_reject = True

        if hasattr(self.forward_kernel, "logpdf") == False:
            raise Exception("Foward kernel has no function called logpdf")
        
        if hasattr(self.forward_kernel, "rvs") == False:
            raise Exception("Foward kernel has no function called rvs")

        if hasattr(self.forward_kernel.momentum_proposal, "logpdf") == False:
            raise Exception("Momentum proposal has no function called logpdf")

        if hasattr(self.forward_kernel.momentum_proposal, "rvs") == False:
            raise Exception("Momentum proposal has no function called rvs")

        # Hold etimated quantities and diagnostic metrics
        if hasattr(self.target, "constrained_dim"):
            self.mean_estimate = np.zeros([self.K + 1, self.target.constrained_dim])
            self.recycled_mean_estimate = np.zeros([self.K + 1, self.target.constrained_dim])
            self.variance_estimate = np.zeros([self.K + 1, self.target.constrained_dim])
            self.recycled_variance_estimate = np.zeros([self.K + 1, self.target.constrained_dim])
        else:
            self.mean_estimate = np.zeros([self.K + 1, self.target.dim])
            self.recycled_mean_estimate = np.zeros([self.K + 1, self.target.dim])
            self.variance_estimate = np.zeros([self.K + 1, self.target.dim])
            self.recycled_variance_estimate = np.zeros([self.K + 1, self.target.dim])
        self.resampled = [False] * (self.K + 1)
        self.ess = np.zeros(self.K + 1)
        self.log_likelihood = np.zeros(self.K + 1)
        self.recycling_constant = np.zeros(self.K + 1)
        self.phi = np.zeros(self.K + 1)
        self.acceptance_rate = np.zeros(self.K)
        self.run_time = None

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
        Resamples samples and their weights from the specified indexes. If running the SMC sampler
        in parallel using MPI, we resample locally on rank zero and then scatter the resampled
        samples to the other ranks.

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

    def sample(
        self,
        save_samples=False,
        show_progress=True,
    ):
        """
        Sample from the target distribution using an SMC sampler.
        """

        start_time = time()

        # Tensors to hold samples drawn at each iteration
        if save_samples:
            self.x_saved = np.zeros([self.K + 1, self.N, self.target.dim])
            self.logw_saved = np.zeros([self.K + 1, self.N])

        # Draw initial samples from the sample proposal distribution
        x = self.sample_proposal.rvs(self.N)
        x_new = np.zeros([self.N, self.target.dim])

        # Tensors to hold momenta, gradients and number of leapfrog steps
        r_new= np.zeros([self.N, self.target.dim])
        grad_x = np.zeros([self.N, self.target.dim])

        # Calculate the initial temperature
        phi_old, phi_new = (0.0, 0.0) if self.tempering else (1.0, 1.0)
        if self.tempering:
            if isinstance(self.tempering, AdaptiveTempering):
                p_logpdf_x_phi = self.target.logpdf(x, phi=phi_old)
                args = [x, p_logpdf_x_phi, phi_old]
            phi_new = self.tempering.calculate_phi(args)
            if self.verbose:
                print(f"Initial temperature: {phi_new}")

        # Calculate the initial weights
        logw = np.zeros(self.N)
        logw_new = np.zeros(self.N)
        p_logpdf_x = self.target.logpdf(x, phi=phi_new)
        q0_logpdf_x = self.sample_proposal.logpdf(x)
        logw = p_logpdf_x - q0_logpdf_x

        # Save initial samples
        if save_samples:
            self.x_saved[0] = x
            self.logw_saved[0] = logw

        # Only create the tqdm progress bar on rank zero
        progress_bar = tqdm(total=self.K, desc=f"NUTS Sampling", disable=not show_progress)

        # Main sampling loop
        for k in range(self.K):
            # Record new temperature
            self.phi[k] = phi_new

            # Normalise importance weights and calculate the log likelihood
            wn, self.log_likelihood[k] = self.normalise_weights(logw)

            if self.recycling:
                # Calculate the recycling constant
                if isinstance(self.recycling, ESSRecycling):
                    self.recycling_constant[k] = self.recycling.constant(wn)

            # Estimate the mean and variance of the target distribution
            self.mean_estimate[k], self.variance_estimate[k] = self.estimate(x, wn)

            # Calculate the effective sample size and resample if necessary
            self.ess[k] = self.calculate_ess(wn)
            if self.ess[k] < self.N / 2:
                if self.verbose:
                    print(f"Resampling iteration {k} with ESS {self.ess[k]}")
                self.resampled[k] = True
                x, logw = self.resample(x, wn, self.log_likelihood[k])

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

            if save_samples:
                self.x_saved[k + 1] = x_new.copy()
                self.logw_saved[k + 1] = logw_new.copy()

            progress_bar.update(1)

        progress_bar.close()

        # Normalise importance weights and calculate the log likelihood
        wn, self.log_likelihood[self.K] = self.normalise_weights(logw)

        if self.recycling:
            # Calculate the recycling constant
            if isinstance(self.recycling, ESSRecycling):
                self.recycling_constant[self.K] = self.recycling.constant(wn)

        # Estimate the mean and variance of the target distribution
        self.mean_estimate[self.K], self.variance_estimate[self.K] = self.estimate(x, wn)

        # Calculate the effective sample size and resample if necessary
        self.ess[self.K] = self.calculate_ess(wn)

        self.phi[self.K] = phi_new

        if self.recycling:
            # Recycle the mean and variance estimates
            self.recycled_mean_estimate = self.recycling.recycle_mean(
                self.mean_estimate, self.recycling_constant
            )
            self.recycled_variance_estimate = self.recycling.recycle_variance(
                self.variance_estimate,
                self.mean_estimate,
                self.recycled_mean_estimate,
                self.recycling_constant,
            )

        self.run_time = time() - start_time
