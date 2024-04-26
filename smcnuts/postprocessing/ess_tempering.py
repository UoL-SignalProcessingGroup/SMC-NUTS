import numpy as np


def estimate_from_tempered(target, smcs):
    """ Calculate adjusted weights, and form estimates of all past simulated samples.

    This function calculates the adjusted importance weights `ess_logw` for all samples. The
    weights are defined as \pi(x) / \pi(x, \phi_k) where \pi(x) is the target density and
    \pi(x, \phi_k) is the density of the kth proposal. The adjusted weights are then used to
    form estimates of the mean and variance of the target density.
    """

    ess_logw = np.zeros([smcs.K+1, smcs.N])

    for k in range(smcs.K+1):
        # Using weights calculated in the the sampler draw a set a set of samples
        smcs.samples.wn = smcs.logw_saved[k]
        smcs.samples.normalise_weights()
        wn = smcs.samples.wn
        
        z = np.linspace(0, smcs.N-1, smcs.N, dtype=int)
        z_new = smcs.rng.choice(z, smcs.N, p=wn)
        x = smcs.x_saved[k].copy()[z_new]

        # Calculate importance weights and normalise
        ess_logw[k] = target.logpdf(x) - target.logpdf(x, phi=smcs.phi[k])
        
        smcs.samples.wn = ess_logw[k]
        smcs.samples.normalise_weights()
        ess_wn = smcs.samples.wn

        # Calculate mean and variance estimates
        smcs.mean_estimate[k], smcs.variance_estimate[k] = smcs.estimate(x, ess_wn)

    return smcs
    
    

