import autograd.numpy as np
from scipy.stats import multivariate_normal
from .utils import hmc_accept_reject

# Set max tree death of NUTS tree, default is 2^10.
MAX_TREE_DEPTH = 10

class NUTSProposal:
    """No-U-Turn Sampler Proposal

    Propagate samples using the proposal from the No-U-Turn proposal [1]. Algorithm is largely based on Alg. 3 of the reference. 


    [1] https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf

    Attributes:
        target: Target distribution of interest.
        dim: Dimensionality of the system.
        momentum_proposal: Momentum proposal distribution.
        step_size: Step size for the leapfrog integrator.
        
    """

    def __init__(
        self,
        target,
        momentum_proposal,
        step_size,
        accept_reject: bool = False,
        rng = np.random.default_rng(),
    ):
        self.target = target
        self.momentum_proposal = momentum_proposal
        self.accept_reject = accept_reject
        self.step_size = step_size
        self.rng = rng
        self.dist = multivariate_normal(np.zeros(self.target.dim), np.eye(self.target.dim), seed=rng)

    def rvs(self, x_cond, r_cond, grad_x, phi: float = 1.0):
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

        x_prime, r_prime = np.zeros_like(x_cond), np.zeros_like(r_cond)

        # For each sample, generate a new set of proposed samples using NUTS
        for i in range(len(x_cond)):
            x_prime[i], r_prime[i] = self.generate_nuts_samples(
                x_cond[i], r_cond[i], grad_x[i], phi=phi
            )

        if self.accept_reject:
            accepted = np.array([False] * len(x_prime))
            for i in range(len(x_prime)):
                accepted[i] = hmc_accept_reject(self.target.logpdf, x_cond[i], x_prime[i], r_cond[i], r_prime[i], rng=self.rng)
            x_prime[~accepted] = x_cond[~accepted]
            r_prime[~accepted] = r_cond[~accepted]

        return x_prime, r_prime

    def generate_nuts_samples(self, x0, r0, grad_x, phi: float = 1.0):

        """
        Description
        -----------
        Generates samples using the NUTS proposal, Based off Alg. 3 in [1]
        """
        
        logp = self.target.logpdf(x0, phi=phi)    
        self.H0 = logp - 0.5 * np.dot(r0, r0.T)            
        logu = float(self.H0 - self.rng.exponential(1))
        
        # initialize the NUTS tree 
        x = x0
        xminus = x0
        xplus = x0
        rminus = r0
        rplus = r0
        r = r0
        gradminus = grad_x
        gradplus = grad_x
 

        depth = 0  
        n = 1  
        stop = 0  

        while (stop == 0):
            # Using a Bernoulli try choose a direction. -1 (backwards) or +1 (forwards)
            direction = int(2 * (self.rng.uniform(0,1) < 0.5) - 1)

            if (direction == -1):
                xminus, rminus, gradminus, _, _, _, xprime, rprime, nprime, stopprime= self.build_tree(xminus, rminus, gradminus, logu, direction, depth,  phi)
            else:
                _, _, _, xplus, rplus, gradplus, xprime, rprime, nprime, stopprime  = self.build_tree(xplus, rplus, gradplus, logu, direction, depth, phi)


            if (stopprime == 0 and self.rng.uniform() < min(1., float(nprime) / float(n))):
                x = xprime
                r = rprime

            n += nprime

            stop = stopprime or self.stop_criterion(xminus, xplus, rminus, rplus)           
            
            depth += 1
            
            if(depth > MAX_TREE_DEPTH):
                break
        
        return x, r

    def build_tree(self, x, r, grad_x, logu, direction, depth, temperature=1.0):
        if (depth == 0):
            xprime, rprime, gradprime = self.NUTSLeapfrog(x, r, grad_x, direction, temperature)
            logpprime = self.target.logpdf(xprime, phi=temperature)
            joint = logpprime - 0.5 * np.dot(rprime, rprime.T)
            nprime = int(logu < joint)
            stopprime = int((logu - 100.) >= joint)
            xminus = xprime
            xplus = xprime
            rminus = rprime
            rplus = rprime
            gradminus = gradprime
            gradplus = gradprime
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.                                                                               
            xminus, rminus, gradminus, xplus, rplus, gradplus, xprime, rprime,  nprime, stopprime = self.build_tree(x, r, grad_x, logu, direction, depth - 1,  temperature)
            
            if (stopprime == 0):
                if (direction == -1):
                    xminus, rminus, gradminus, _, _, _, xprime2, rprime2, nprime2, stopprime2  = self.build_tree(xminus, rminus, gradminus, logu, direction, depth - 1,  temperature)
                else:
                    _, _, _, xplus, rplus, gradplus, xprime2, rprime2, nprime2, stopprime2   = self.build_tree(xplus, rplus, gradplus, logu, direction, depth - 1, temperature)           
               
                if (self.rng.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                    xprime = xprime2
                    rprime = rprime2

                nprime = int(nprime) + int(nprime2)

                stopprime = int(stopprime or stopprime2 or self.stop_criterion(xminus, xplus, rminus, rplus))

        return xminus, rminus, gradminus, xplus, rplus, gradplus, xprime, rprime,  nprime, stopprime

    def stop_criterion(self, xminus, xplus, rminus, rplus):
        """
        Description
        -----------
        Checks if a U-turn is present in the furthest nodes in the NUTS
        tree
        """
        dx = xplus - xminus
        return (np.dot(dx, rminus.T) < 0) or (np.dot(dx, rplus.T) < 0)

    def NUTSLeapfrog(self, x, r, grad_x, direction, temperature=1.0):
    
        """
        Description
        -----------
        Performs a single Leapfrog step returning the final position, momentum and gradient.
        """
        r = np.add(r, (direction*self.step_size/2)*grad_x)
        x = np.add(x, direction*self.step_size*r)
        grad_x = self.target.logpdfgrad(x, phi=temperature)
  
        if temperature is not None:
            grad_x *= temperature

        r = np.add(r, (direction*self.step_size/2)*grad_x)
        
        return x, r, grad_x

    def logpdf(self, r):
        """
        Description:
            Calculate the log probability of the forward kernel, i.e. the momentum.

        Args:
            r: Particle momenta.

        Returns:
            log_prob: Log probability of the forward kernel.
        """

        return self.dist.logpdf(r)
