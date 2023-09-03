import autograd.numpy as np
from scipy.stats import multivariate_normal

from .utils import hmc_accept_reject

MAX_TREE_DEPTH = 10


"""
THIS CLASS IS A MERGE OF THE HMCPROPOSAL CLASS AND THE NUTS PROPSAL FROM
THE BIG HYPOTHESES PYTHON REPOSITORY. IT IS NOT YET FINISHED AND NEEDS TIDYING
UP AND TESTING.

CURRENTLY RUNS SEQUENTIALLY AND IS NOT VECTORISED.
"""

class NUTSProposal:
    """Hamiltonian Monte Carlo Proposal

    Propagate samples using a Hamiltonian Monte Carlo (HMC) proposal. HMC propagates
    samples using an approximate simulation of the Hamiltonian dynamics of a system.

    [1] https://mc-stan.org/docs/2_19/reference-manual/hamiltonian-monte-carlo.html

    Attributes:
        dim: Dimensionality of the system.
        target: Target distribution of interest.
        step_size: Step size for the leapfrog integrator.
        num_steps: Number of leapfrog steps to take.
        momentum_proposal: Momentum proposal distribution.
        random_num_steps: Whether or not to use a random number of leapfrog steps.
    """

    def __init__(
        self,
        target,
        momentum_proposal,
        integrator,
        accept_reject: bool = False,
        rng = np.random.default_rng(),
    ):
        self.target = target
        self.momentum_proposal = momentum_proposal
        self.integrator = integrator
        self.accept_reject = accept_reject
        self.rng = rng

        self.dist = multivariate_normal(np.zeros(self.target.dim), np.eye(self.target.dim), seed=rng)

    def rvs(self, x_cond, v_cond, grad_x, phi: float = 1.0):
        """
        Description:
            Propogate a set of samples using Hamiltonian Monte Carlo (HMC).

        Args:
            x_cond: Current particle positions.
            v_cond: Current particle velocities.
            grad_x: Current particle gradients.

        Returns:
            x_prime: Updated particle positions.
            v_prime: Updated particle velocities.
            T: Length of Leapfrog trajectories.
        """

        x_prime, v_prime = np.zeros_like(x_cond), np.zeros_like(v_cond)
        T = np.zeros(len(x_cond))

        for i in range(len(x_cond)):
            x_prime[i], v_prime[i], T[i] = self.generate_nuts_samples(
                x_cond[i], v_cond[i], grad_x[i], phi=phi
            )

        if self.accept_reject:
            accepted = np.array([False] * len(x_prime))
            for i in range(len(x_prime)):
                accepted[i] = hmc_accept_reject(self.target.logpdf, x_cond[i], x_prime[i], v_cond[i], v_prime[i], rng=self.rng)
            x_prime[~accepted] = x_cond[~accepted]
            v_prime[~accepted] = v_cond[~accepted]

        return x_prime, v_prime, T

    def generate_nuts_samples(self, x0, v0, grad_x, phi: float = 1.0):

        """
        Description
        -----------
        Checks if a U-turn is present in the furthest nodes in the NUTS
        tree
        """
        
        #joint lnp of x and momentum r
        logp = self.target.logpdf(x0, phi=phi)    
        self.H0 = logp - 0.5 * np.dot(v0, v0.T)            
        logu = float(self.H0 - self.rng.exponential(1))
        
        # initialize the tree 
        x = x0
        xminus = x0
        xplus = x0
        vminus = v0
        vplus = v0
        v = -v0
        gradminus = grad_x
        gradplus = grad_x
        t=0
        tplus=t
        tminus=t

        depth = 0  # initial depth of the tree
        n = 1  # Initially the only valid point is the initial point.
        stop = 0  # Main loop: will keep going until stop == 1.

        while (stop == 0):
            # Choose a direction. -1 = backwards, 1 = forwards.
            direction = int(2 * (self.rng.uniform(0,1) < 0.5) - 1)

            if (direction == -1):
                xminus, vminus, gradminus, _, _, _, xprime, vprime, logpprime, nprime, stopprime, tminus, _, tprime = self.build_tree(xminus, vminus, gradminus, logu, direction, depth, tminus, phi)
            else:
                _, _, _, xplus, vplus, gradplus, xprime, vprime, logpprime, nprime, stopprime, _, tplus, tprime    = self.build_tree(xplus, vplus, gradplus, logu, direction, depth, tplus, phi)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            if (stopprime == 0 and self.rng.uniform() < min(1., float(nprime) / float(n))):
                x = xprime
                v = vprime
                t = tprime

            # Update number of valid points we've seen.
            n += nprime

            # Decide if it's time to stop.
            stop = stopprime or self.stop_criterion(xminus, xplus, vminus, vplus)           
            
            # Increment depth.
            depth += 1
            
            if(depth > MAX_TREE_DEPTH):
                # print("Max tree size in NUTS reached")
                break
        
        return x, v, t

    def build_tree(self, x, v, grad_x, logu, direction, depth, t, temperature=1.0):
        if (depth == 0):
            xprime, vprime, gradprime = self.NUTSLeapfrog(x, v, grad_x, direction, temperature)
            logpprime = self.target.logpdf(xprime, phi=temperature)
            joint = logpprime - 0.5 * np.dot(vprime, vprime.T)
            nprime = int(logu < joint)
            stopprime = int((logu - 100.) >= joint)
            xminus = xprime
            xplus = xprime
            vminus = vprime
            vplus = vprime
            gradminus = gradprime
            gradplus = gradprime
            tprime = t + self.integrator.step_size
            tminus=tprime
            tplus=tprime
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.                                                                               
            xminus, vminus, gradminus, xplus, vplus, gradplus, xprime, vprime, logpprime, nprime, stopprime, tminus, tplus, tprime  = self.build_tree(x, v, grad_x, logu, direction, depth - 1, t, temperature)
            
            # No need to keep going if the stopping criteria were met in the first subtree.
            if (stopprime == 0):
                if (direction == -1):
                    xminus, vminus, gradminus, _, _, _, xprime2, vprime2, logpprime2, nprime2, stopprime2, tminus, _,  tprime2    = self.build_tree(xminus, vminus, gradminus, logu, direction, depth - 1, tminus, temperature)
                else:
                    _, _, _, xplus, vplus, gradplus, xprime2, vprime2, logpprime2, nprime2, stopprime2, _, tplus, tprime2       = self.build_tree(xplus, vplus, gradplus, logu, direction, depth - 1, tplus, temperature)           
               
                if (self.rng.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                    xprime = xprime2
                    logpprime = logpprime2
                    vprime = vprime2
                    tprime=tprime2

                # Update the number of valid points.
                nprime = int(nprime) + int(nprime2)

                # Update the stopping criterion.
                stopprime = int(stopprime or stopprime2 or self.stop_criterion(xminus, xplus, vminus, vplus))

        return xminus, vminus, gradminus, xplus, vplus, gradplus, xprime, vprime, logpprime, nprime, stopprime, tminus, tplus, tprime

    def stop_criterion(self, xminus, xplus, rminus, rplus):
        """
        Description
        -----------
        Checks if a U-turn is present in the furthest nodes in the NUTS
        tree
        """
        dx = xplus - xminus
        return (np.dot(dx, rminus.T) < 0) or (np.dot(dx, rplus.T) < 0)

    def NUTSLeapfrog(self, x, v, grad_x, direction, temperature=1.0):
    
        """
        Description
        -----------
        Performs a single Leapfrog step returning the final position, velocity and gradient.
        """
        v = np.add(v, (direction*self.integrator.step_size/2)*grad_x)
        x = np.add(x, direction*self.integrator.step_size*v)
        grad_x = self.target.logpdfgrad(x, phi=temperature)
  
        if temperature is not None:
            grad_x *= temperature

        v = np.add(v, (direction*self.integrator.step_size/2)*grad_x)
        
        return x, v, grad_x

    def logpdf(self, v):
        """
        Description:
            Calculate the log probability of the forward kernel.

        Args:
            v: Particle velocities.

        Returns:
            log_prob: Log probability of the forward kernel.
        """

        return self.dist.logpdf(v)
