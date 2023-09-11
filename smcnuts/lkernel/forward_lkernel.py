import numpy as np
from autograd.scipy import stats as AutoStats


class ForwardLKernel:
    """Forward Kernel L Kernel

    The forward kernel approximation of the optimal L-kernel is
    presented in [1].

    [1] Devlin, L., Horridge, P., Green, P. and Maskell, S (2021). The
    No-U-Turn Sampler as a Proposal Distribution in a Sequential Monte
    Carlo Sampler with a Near-Optimal L Kernel.

    Attributes:
        target: Target distribution.
    """

    def __init__(self, target):
        self.target = target

    def calculate_L(self, x, x_new, v, v_new):
        """
        Description:
            Calculate the Forward Kernel approximation of the optimal L-kernel
            for a Hamiltonian Monte Carlo (HMC) proposal.

        Args:
            x: Current particle positions.
            x_new: New particle positions.
            v: Current particle velocities.
            v_new: New particle velocities.

        Returns:
            log_pdf: The forward kernel approximation of the optimal L-kernel.
        """


        return AutoStats.multivariate_normal.logpdf(np.multiply(-1, v_new), mean=np.zeros(self.target.dim), cov=np.eye(self.target.dim))
