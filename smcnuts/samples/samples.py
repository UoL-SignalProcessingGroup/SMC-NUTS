import numpy as np
from scipy.special import logsumexp
from smcnuts.lkernel.forward_lkernel import ForwardLKernel
from smcnuts.lkernel.gaussian_lkernel import GaussianApproxLKernel
from smcnuts.tempering.adaptive_tempering import ESSTempering

class Samples:
    def __init__(self,
                N,
                D,
                sample_proposal,
                target,
                forward_kernel,
                lkernel,
                tempering,
                rng) -> None:

        """
        Description: Samples is an object that contains the set of SMC samples and their associated methods.

        params:
        N: The number of samples.
        D: The number of dimensions the samples move in.
        sample_proposal: The distribution from which the samples are initially drawn from the problem-space
        target: The target we wish to evaluate
        forward_kernel: The forward proposal (i.e. NUTS)
        lkernel: The L-kernel strategy being used, either i) GaussianApprox ii) forwards iii) asymptotic
        tempering: Boolean. True if target is to be tempered with ESS cooling strategy
        rng: random number seed
        """
        self.N = N
        self.D = D
        self.sample_proposal=sample_proposal
        self.forward_kernel = forward_kernel
        self.target=target
        self.rng = rng

        ## Set-up l-kernel if it is a function to be evaluated
        if lkernel == "GaussianApproxLKernel":
            self.lkernel = GaussianApproxLKernel(target=self.target, N=self.N)
            self.reweight_strategy = self._non_asympototic_reweight
        elif lkernel == "forwardsLKernel":
            self.lkernel = ForwardLKernel(target=self.target, momentum_proposal=self.forward_kernel.momentum_proposal)
            self.reweight_strategy = self._non_asympototic_reweight
        elif lkernel == "asymptoticLKernel":
            self.reweight_strategy = self._asymptotic_reweight
        else:
            raise Exception("Unknown L-kernel supplied") 

        # Set up tempering if it is being used, if not set all the temperatures to be equal to 1.0  
        if tempering:
            self.TemperingScheme = ESSTempering(self.N, self.target, alpha = 0.5)
            self.update_temperature = self._tempering
            self.phi_old = 0.0
            self.phi_new = 0.0
        else:
            self.TemperingScheme = None
            self.update_temperature =  lambda : 1.0
            self.phi_old = 1.0
            self.phi_new = 1.0
        
        # Set up initial sample properties
        self.initialise_samples()
         

    def initialise_samples(self):
        """
        Description: Initialise the properties of the samples and allocate arrarys
        
        x: The location of samples in the target space
        x_new: The location of samples in the target space after a proposal step 
        ess: The number of effective samples
        r: The momentum at the start of a proposal
        r_new: The momentum at the end of a proposal
        phi_new: The new temperature of the system, calculated from phi_old
        logw: sample weights in log space calculated by \pi(x)-q(x), where q is the initial sample proposal
        logw_new: sample weights in log space after a proposal
        wn: Vector of normalised weights
        """
        self.x = self.sample_proposal.rvs(self.N)
        self.x_new = np.copy(self.x)
        self.ess = 0
        self.r= np.zeros([self.N, self.D])
        self.r_new= np.zeros([self.N, self.D])
        self.phi_new = self.update_temperature()
        self.phi_old = self.phi_new

        self.logw = self.target.logpdf(self.x, phi=self.phi_new) - self.sample_proposal.logpdf(self.x)
        
        self.logw_new = np.zeros([self.N])
        self.wn = np.zeros([self.N])

    
    def normalise_weights(self):
        """
        Description: Normalises the sample weights in log scale
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
        Description: Calculate the effective sample size using the normalised
        sample weights.
        """
        self.ess = 1 / np.sum(np.square(self.wn))


    def resample_if_required(self):
        """
        Description: Resample if effective sample size is below the threshold (hard coded to 1/2)
        """
        if(self.ess < self.N / 2):
            self._resample(self.x,  self.wn, self.log_likelihood)


    def _resample(self, x, wn, log_likelihood):
        """
        Description: Resamples samples and their weights from the specified indexes.

        Args:
            x: A list of samples to resample
            wn: A list of normalise sample weights to resample

        Returns:
            x_new: A list of resampled samples
            logw_new: A list of resampled weights
        """

        # Resample x
        i = np.linspace(0, self.N-1, self.N, dtype=int)
        i_new = self.rng.choice(i, self.N, p=wn)
        x_new = x[i_new]

        # Determine new weights
        logw_new = (np.ones(self.N) * log_likelihood) - np.log(self.N)

        self.x = x_new
        self.logw = logw_new    


    def propose_samples(self):
        """
        Description: Run proposal distribution to generate a new set of samples
        
        """
        # Sample initial momentum
        self.r = self.forward_kernel.momentum_proposal.rvs(self.N)
   	
   	    # Propogate particles through the forward kernel
        self.x_new, self.r_new= self.forward_kernel.rvs(self.x, self.r, phi=self.phi_new)
        
        
    def reweight(self):
        """
        Description: Calculate the new weights dependding on the reweighting strategy being used
        
        """
        self.logw_new = self.reweight_strategy()

            
    def _asymptotic_reweight(self):
        """
        Description: Reweight strategy for the asymptotic L-kernel. Note, that it is assumed that this strategy will use tempering

        returns:
        New weights
        
        """
        p_logpdf_x_phi_old = self.target.logpdf(self.x, phi=self.phi_old)
        p_logpdf_x_phi_new = self.target.logpdf(self.x, phi=self.phi_new)

        return self.logw + p_logpdf_x_phi_new - p_logpdf_x_phi_old
        

    def _non_asympototic_reweight(self):
        """
        Description: Reweight strategy for the non-asymptotic L-kernel 

        returns:
        New weights
        """
        p_logpdf_x = self.target.logpdf(self.x)
        p_logpdf_xnew = self.target.logpdf(self.x_new)

        lkernel_logpdf = self.lkernel.calculate_L(self.r_new, self.x_new)
        q_logpdf = self.forward_kernel.logpdf(self.r)

        return self.logw + p_logpdf_xnew - p_logpdf_x + lkernel_logpdf - q_logpdf
    
    
    def _tempering(self):
        """
        Description: Reweight strategy for the asymptotic L-kernel 

        returns:
        New weights
        
        """
        p_logpdf_x_new_phi_old = self.target.logpdf(self.x_new, phi=self.phi_old)
        args = [self.x_new, p_logpdf_x_new_phi_old, self.phi_old]        
        phi_new = self.TemperingScheme.calculate_phi(args)
        self.phi_new = phi_new
        
        return phi_new
        
    
    def update_samples(self):
        """
        Description: Update the samples for the next iteration
        
        """
        self.phi_old = self.phi_new
        self.x = self.x_new
        self.logw = self.logw_new
