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

    def calculate_L(self, r_new, _ ):
        """
        Description:
            Calculate the Forward Kernel approximation of the L-kernel
            for a Hamiltonian Monte Carlo (HMC)-based proposal.

        Args:
            r_new: New particle momenta.

        Returns:
            log_pdf: The forward kernel approximation of the optimal L-kernel.
        """


        return AutoStats.multivariate_normal.logpdf(np.multiply(-1, r_new), mean=np.zeros(self.target.dim), cov=np.eye(self.target.dim))
