from time import time
import numpy as np
from tqdm import tqdm
from smcnuts.samples.samples import Samples
from smcnuts.proposal.nuts_acc_rej import NUTSProposalWithAccRej
from smcnuts.proposal.nuts import NUTSProposal
from smcnuts.estimate.estimate import Estimate
from smcnuts.estimate.estimate_from_tempered import EstimateFromTempered


class SMCSampler():
    """Hamiltonian Monte Carlo (NUTS) SMC Sampler

    Description: An SMC sampler that uses Hamiltonian Monte Carlo (HMC) methods to sample
    from the target distribution of interest.

    Attributes:
        K: Number of iterations.
        N: Number of particles.
        target: Target distribution of interest.
        sample_proposal: Distribution to draw the initial samples from (q0).
        lkernel: Approximation method for the optimum L-kernel.
    """
    
    def __init__(
        self,
        K: int,
        N: int,
        target,
        step_size: float,
        sample_proposal,
        momentum_proposal,
        lkernel,
        tempering=False,
        rng = np.random.default_rng(),
    ):
        self.K = K  # Number of iterations
        self.N = N  # Number of particles
        self.target = target  # Target distribution
        self.rng=rng
        self.lkernel = lkernel


        # Force asymptotic forward kernels to use NUTS with accept-reject mechanism with tempering
        if(lkernel=="asymptoticLKernel"):
            forward_kernel = NUTSProposalWithAccRej(
            target=self.target,
            momentum_proposal=momentum_proposal,
            step_size = step_size,
            rng=self.rng)

            self.estimator = EstimateFromTempered(self.target, self.N, self.K, self.rng)
            

        else:
            forward_kernel = NUTSProposal(
            target=self.target,
            momentum_proposal=momentum_proposal,
            step_size = step_size,
            rng=self.rng)

            self.estimator = Estimate(self.target)

            
        # Set up arrays to be output when the sampler has finished
        self.resampled = [False] * (self.K + 1)
        self.ess = np.zeros(self.K + 1)
        self.log_likelihood = np.zeros(self.K + 1)
        self.phi = np.zeros(self.K + 1)
        self.acceptance_rate = np.zeros(self.K+1)
        self.run_time = None

        self.x_saved = np.zeros([self.K + 1, self.N, self.target.dim])
        self.logw_saved = np.zeros([self.K + 1, self.N])

        # Generate the set of samples to be used in the sampling process
        self.samples = Samples(self.N, self.target.dim, sample_proposal, self.target, forward_kernel, lkernel, tempering, rng) 
        self.samples.initialise_samples()

        self.x_saved[0] = self.samples.x
        self.logw_saved[0] = self.samples.logw

        # Create arrray for mean and variance estimates
        self.mean_estimate = np.zeros([self.K + 1, self.target.dim])
        self.variance_estimate = np.zeros([self.K + 1, self.target.dim])


    def update_sampler(self, k, mean_estimate, variance_estimate):
        """
            Description: Update the sampler for evaluation purposes to output.
        """
        
        self.log_likelihood[k] = self.samples.log_likelihood
        self.mean_estimate[k] = mean_estimate
        self.variance_estimate[k] = variance_estimate
        self.ess[k] = self.samples.ess
        self.acceptance_rate[k] = (np.sum(np.all(self.samples.x_new != self.samples.x, axis=1)) / self.N) # Calculate number of accepted particles
 


    def sample(self, show_progress=True):
        """
        Description: Sample from the target distribution using an SMC sampler.
        """

        start_time = time()

        # Main sampling loop
        for k in tqdm(range(self.K), desc=f"NUTS Sampling", disable=not show_progress):
            #update temperature
            self.phi[k] = self.samples.phi_new

            # Normalise the weights
            self.samples.normalise_weights()

            # Form estimates
            mean_estimate, variance_estimate = self.estimator.return_estimate(self.samples.x, self.samples.wn)
            
            # Calculate the effective sample size
            self.samples.calculate_ess()

            # Resample if necessary
            self.samples.resample_if_required()

            # Propose new samples
            self.samples.propose_samples()

            # Temper distribution (non-tempered setting will result with \phi always equal to 1.0)
            self.samples.update_temperature()
            
            # Reweight samples
            self.samples.reweight()
            
            # Update sampler properties for current iteration
            self.update_sampler(k, mean_estimate, variance_estimate)
            
            # Update samples at the end of current iteration
            self.samples.update_samples()
            self.x_saved[k+1] = self.samples.x_new
            self.logw_saved[k+1] = self.samples.logw_new
            
        # Calculate the final params based on the final proposal step  
        self.samples.normalise_weights()
        mean_estimate, variance_estimate = self.estimator.return_estimate(self.samples.x, self.samples.wn)
        self.samples.calculate_ess()

        # Update sampler properties for the final proposal step
        self.update_sampler(self.K, mean_estimate, variance_estimate)
        self.phi[self.K] = self.samples.phi_new
        
        # If using the asymptotic approach we calculate the estimates using the adjusted tempered weights
        if(self.lkernel=="asymptoticLKernel"):
            self.mean_estimate, self.variance_estimate = self.estimator.estimate_from_tempered(self.x_saved, self.logw_saved, self.phi)

        self.run_time = time() - start_time
