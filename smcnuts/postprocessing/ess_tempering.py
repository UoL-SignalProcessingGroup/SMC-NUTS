import numpy as np


def estimate_and_recycle(target, smcs):
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
