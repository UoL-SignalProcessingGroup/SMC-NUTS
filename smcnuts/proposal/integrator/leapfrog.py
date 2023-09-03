import autograd.numpy as np
from scipy.stats import multivariate_normal


class LeapfrogIntegrator:
    """Leapfrog integrator for continuous target distributions.

    Simulate the Hamiltonian dynamics of a set of particles using
    the Leapfrog integrator. See [1] for more details.

    [1] https://mc-stan.org/docs/2_19/reference-manual/sampling.html

    Attributes:
        target: Target distribution.
        step_size: Step size of leapfrog integrator.
    """

    def __init__(self, target, step_size: float):
        self.target = target
        self.step_size = step_size

    def step(self, x_cond, v_cond, grad_x, phi: float = 1.0):
        """
        Description:
            Perform a single Leapfrog step.

        Args:
            x_cond: Current particle positions.
            v_cond: Current particle velocities.
            grad_x: Current particle gradients.
            phi: Temperature of the target distribution.

        Returns:
            x_prime: Updated particle positions.
            v_prime: Updated particle velocities.
            grad_x_prime: Updated particle gradients.
        """

        v_prime = np.add(v_cond, (self.step_size / 2) * grad_x)
        x_prime = np.add(x_cond, self.step_size * v_prime)
        grad_x = phi * self.target.logpdfgrad(x_prime)
        v_prime = np.add(v_prime, (self.step_size / 2) * grad_x)

        return x_prime, v_prime, grad_x  # type: ignore
