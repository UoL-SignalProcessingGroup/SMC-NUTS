import autograd.numpy as np
from scipy.stats import multivariate_normal


def hmc_accept_reject(target_lpdf, x, x_prime, v, v_prime, phi=1.0, rng=np.random.default_rng()):
    with np.errstate(all='ignore'):
        U = target_lpdf(x_prime, phi=phi) - (0.5 * np.dot(v_prime, v_prime))
        K = target_lpdf(x, phi=phi) - (0.5 * np.dot(v, v))
        acceptance_ratio = np.exp(U - K)
        acceptance_probability = min(1., acceptance_ratio)

        if rng.uniform() >  acceptance_probability or np.any(np.isinf(x_prime)):
            return False
        return True
