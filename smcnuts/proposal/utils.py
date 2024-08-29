import numpy as np

def hmc_accept_reject(target_lpdf, x, x_prime, r, r_prime, phi=1.0, rng=np.random.default_rng()):
    
    """ Calculate whether to accept or reject a move 
        Returns a Boolean of True if the move is accepted and False if rejected.

        Input:
        target_lpdf: Target log pdf 
        x: Intial state
        x_prime: Proposed state
        r: Initial momentum
        r_prime: Proposed Momentum
        phi: Temperature (default = 1.0)
        rng: Random number generator

        Output:
        Boolean: True if accepted, False if rejected.

    """

    with np.errstate(all='ignore'):
        # Calculate the (log) Hamiltonians at the end and start of the proposal
        H1 = target_lpdf(x_prime, phi=phi) - (0.5 * np.dot(r_prime, r_prime))
        H0 = target_lpdf(x, phi=phi) - (0.5 * np.dot(r, r))
        
        # Calculate the acceptanace rate and probability
        acceptance_ratio = np.exp(H1 - H0)
        acceptance_probability = min(1., acceptance_ratio)

        # Generate a random nuber from a uniform distribution and compare to acceptance probability
        if rng.uniform() >  acceptance_probability or np.any(np.isinf(x_prime)):
            return False
        return True
