import numpy as np


def estimate_and_recycle(target, smcs):
    """ Calculate adjusted weights, form estimates and recycle all past simulated samples.

    This function calculates the adjusted importance weights `ess_logw` for all samples. The
    weights are defined as \pi(x) / \pi(x, \phi_k) where \pi(x) is the target density and
    \pi(x, \phi_k) is the density of the kth proposal. The adjusted weights are then used to
    form estimates of the mean and variance of the target density, calculate recycling constants
    and recycle all past simulated samples. This scheme is outlined in [1].

    [1] Nguyen, T., Septier, F., Peters, G. and Delignon, Y. (2014). Improving
    SMC sampler estimate by recycling all past simulated samples.
    """
    ess_logw = np.zeros([smcs.K+1, smcs.N])
    recycling_consntants = np.zeros([smcs.K+1, smcs.N])
    for k in range(smcs.K+1):
        wn, _ = smcs.normalise_weights(smcs.logw_saved[k])
        z = np.linspace(0, smcs.N-1, smcs.N, dtype=int)
        z_new = smcs.rng.choice(z, smcs.N, p=wn)
        x = smcs.x_saved[k].copy()[z_new]

        ess_logw[k] = target.logpdf(x) - target.logpdf(x, phi=smcs.phi[k])
        ess_wn, _ = smcs.normalise_weights(ess_logw[k])
        smcs.mean_estimate[k], smcs.variance_estimate[k] = smcs.estimate(x, ess_wn)
        recycling_consntants[k] = smcs.recycling.constant(ess_wn)

    return smcs
