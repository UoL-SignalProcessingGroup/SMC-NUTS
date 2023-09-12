import numpy as np
import warnings
from autograd.scipy import stats as AutoStats


class GaussianApproxLKernel:
    """Gaussian L Kernel

    The Gaussian approximation of the optimal L-kernel presented
    in [1]. This implementation is based on the code presented in [2].

    [1] Green, P., Moore, R., Jackson, R., Li, J. and Maskell, S. (2020).
    Increasing the efficiency of Sequential Monte Carlo samplers through the
    use of approximately optimal L-kernels.

    [2] https://github.com/plgreenLIRU/SMC_approx_optL

    Attributes:
        D: Dimensionality of the target distribution
    """

    def __init__(self, target, N: int):
        self.D = target.dim
        self.N = N

    def calculate_L(self, r_new, x_new):
        """
        Description:
            Calculate the Forward Kernel approximation of the optimal L-kernel
            for a Hamiltonian Monte Carlo (HMC) proposal.

        Args:
            r_new: New particle momentum.
            x_new: New particle positions.
            

        Returns:
            log_pdf: The forward kernel approximation of the optimal L-kernel.

        Todo:
            Vectorize this function.
        """

        lkernel_pdf = np.zeros(x_new.shape[0])

        # Collect r_new and x_new together into X
        X = np.hstack([-r_new, x_new])

        # Directly estimate the mean and covariance matrix of X
        mu_X = np.mean(X, axis=0)
        cov_X = np.cov(np.transpose(X))

        # Find mean of the joint distribution (p(v_-new, x_new))
        mu_negvnew, mu_xnew = mu_X[0:self.D], mu_X[self.D:2 * self.D]

        # Find covariance matrix of joint distribution (p(-r_new, x_new))
        (cov_negvnew_negv,
            cov_negvnew_xnew,
            cov_xnew_negvnew,
            cov_xnew_xnew) = (cov_X[0:self.D, 0:self.D],
                            cov_X[0:self.D, self.D:2 * self.D],
                            cov_X[self.D:2 * self.D, 0:self.D],
                            cov_X[self.D:2 * self.D, self.D:2 * self.D])

        # Define new L-kernel
        def L_logpdf_vnew(negvnew, x_new):

            # Mean of approximately optimal L-kernel
            mu = (mu_negvnew + cov_negvnew_xnew @ np.linalg.pinv(cov_xnew_xnew) @
                    (x_new - mu_xnew))

            # Variance of approximately optimal L-kernel
            cov = (cov_negvnew_negv - cov_negvnew_xnew @
                    np.linalg.pinv(cov_xnew_xnew) @ cov_xnew_negvnew)

            # Add ridge to avoid singularities
            cov += np.eye(self.D) * 1e-6

            # Log det covariance matrix
            sign, logdet = np.linalg.slogdet(cov)
            log_det_cov = sign * logdet

            # Inverse covariance matrix
            inv_cov = np.linalg.inv(cov)

            # Find log pdf
            logpdf = (-0.5 * log_det_cov -
                        0.5 * (negvnew - mu).T @ inv_cov @ (negvnew - mu))

            return logpdf

        for i in range(x_new.shape[0]):
            lkernel_pdf[i] = L_logpdf_vnew(-r_new[i], x_new[i])

        return lkernel_pdf  # type: ignore
