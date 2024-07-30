import numpy as np
from smcnuts.estimate.estimate import Estimate
from smcnuts.samples.samples import Samples


class EstimateFromTempered(Estimate):

    """
    A derived class from Estimate. This class allows mean and variance estimates to be calculated from a tempered distribution where
    the cooling parameter phi has been calculated within the SMC sampler. 

    """

    def __init__(self, target, N, K, rng):
        super().__init__(target)
        self.N = N
        self.K = K
        self.rng = rng

        # Instanstiate a samples class to calculate required parameters
        self.samples = Samples(N, target.dim, None, target,
                               None, "asymptoticLKernel", True, self.rng)

    def estimate_from_tempered(self, x_saved, logw_saved, phi):
        """ Calculate adjusted weights, and form estimates of all past simulated samples.

        Description: This function calculates the adjusted importance weights `ess_logw` for all samples. The
        weights are defined as \pi(x) / \pi(x, \phi_k) where \pi(x) is the target density and
        \pi(x, \phi_k) is the density of the kth proposal. The adjusted weights are then used to
        form estimates of the mean and variance of the target density.
        """

        mean_estimate = np.zeros([self.K + 1, self.target.dim])
        var_estimate = np.zeros([self.K + 1, self.target.dim])

        for k in range(self.K+1):
            # Using weights calculated in the the sampler draw a set a set of samples
            self.samples.logw = logw_saved[k]
            self.samples.normalise_weights()
            wn = self.samples.wn

            z = np.linspace(0, self.N-1, self.N, dtype=int)
            z_new = self.rng.choice(z, self.N, p=wn)
            x = x_saved[k].copy()[z_new]

            # Calculate importance weights and normalise
            self.samples.logw = self.target.logpdf(x) - self.target.logpdf(x, phi=phi[k])
             
            self.samples.normalise_weights()
            ess_wn = self.samples.wn

            # Calculate mean and variance estimates
            mean_estimate[k], var_estimate[k] = self.return_estimate(x, ess_wn)
        
        return mean_estimate, var_estimate
