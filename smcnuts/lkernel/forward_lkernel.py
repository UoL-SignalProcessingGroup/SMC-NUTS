import numpy as np


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

    def __init__(self, target, momentum_proposal):
        self.target = target
        self.momentum_proposal = momentum_proposal

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

        return self.momentum_proposal.logpdf(np.multiply(-1, r_new))
    
    def return_weight(self, logw, x, x_new):
            p_logpdf_x = self.target.logpdf(x)
            p_logpdf_xnew = self.target.logpdf(x_new)

            lkernel_logpdf = self.calculate_L(r_new, x_new)
            q_logpdf = self.forward_kernel.logpdf(r)

            logw_new = (
                logw + p_logpdf_xnew - p_logpdf_x + lkernel_logpdf - q_logpdf
            )