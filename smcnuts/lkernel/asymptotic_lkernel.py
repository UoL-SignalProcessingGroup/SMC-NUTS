import numpy as np

class AsymptoticLKernel:
    """Asymptotic L-kernel class

    Attributes:
        target: Target distribution.
    """

    def __init__(self, target, momentum_proposal):
        self.target = target
        self.momentum_proposal = momentum_proposal
        self.phi_old = 0
        self.phi_new = 0 
        
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
    
    def return_weight(self, logw, x, _):
        p_logpdf_x_phi_old = self.target.logpdf(x, phi=self.phi_old)
        p_logpdf_x_phi_new = self.target.logpdf(x, phi=self.phi_new)

        return logw + p_logpdf_x_phi_new - p_logpdf_x_phi_old

    
