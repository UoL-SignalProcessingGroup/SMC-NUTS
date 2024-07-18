import autograd.numpy as np
from scipy.stats import multivariate_normal
from .utils import hmc_accept_reject
from smcnuts.proposal.nuts import NUTSProposal

# Set max tree death of NUTS tree, default is 2^10.
MAX_TREE_DEPTH = 10

class NUTSProposalWithAccRej(NUTSProposal):
    """No-U-Turn Sampler Proposal

    Propagate samples using the proposal from the No-U-Turn proposal [1] with accept-reject step at the end. Algorithm is largely based on Alg. 3 of the reference. 


    [1] https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf

    Attributes:
        target: Target distribution of interest.
        dim: Dimensionality of the system.
        momentum_proposal: Momentum proposal distribution.
        step_size: Step size for the leapfrog integrator.
        
    """
    
    def __init__(self, target, momentum_proposal, step_size, rng = np.random.default_rng()):
        super().__init__(target,  momentum_proposal, step_size, rng)

    def rvs(self, x_cond, r_cond, phi: float = 1.0):
        """
        Description:
            Propogate a set of samples using the proposal from the No-U-Turn Sampler.

        Args:
            x_cond: Current particle positions.
            r_cond: Current particle momenta.
            grad_x: Current particle gradients.

        Returns:
            x_prime: Updated particle positions.
            r_prime: Updated particle momenta.
        """

        x_prime, r_prime = super(NUTSProposalWithAccRej, self).rvs(x_cond, r_cond, phi)
   
        # Apply an accept-reject step for the assymptoptic L-kernel. 
        accepted = np.array([False] * len(x_prime))
        for i in range(len(x_prime)):
            accepted[i] = hmc_accept_reject(self.target.logpdf, x_cond[i], x_prime[i], r_cond[i], r_prime[i], phi,rng=self.rng)
        x_prime[~accepted] = x_cond[~accepted]
        r_prime[~accepted] = r_cond[~accepted]
    

        return x_prime, r_prime
