import warnings
import numpy as np

"""
Here we set some functions which are used as prechecks before the SMC sampler is run.

"""


def Check_Fwd_Proposal(FwdProposalInstance):
    """
    Description:
        Raises an exception if the forwards proposal does not have the required attributes to perform SMC.

    Args:
        FwdProposalInstance: Instance of a forwards propoposal to be used.

    """
    if hasattr(FwdProposalInstance, "logpdf") == False:
        raise Exception("Foward kernel has no function called logpdf")
    
    if hasattr(FwdProposalInstance, "rvs") == False:
        raise Exception("Foward kernel has no function called rvs")

    if hasattr(FwdProposalInstance.momentum_proposal, "logpdf") == False:
        raise Exception("Momentum proposal has no function called logpdf")

    if hasattr(FwdProposalInstance.momentum_proposal, "rvs") == False:
        raise Exception("Momentum proposal has no function called rvs")

    return


def Check_Asym_Has_AccRej(SMC):
    """
    Description:
        If the asymptotic L-kernel is being used, then using accept-reject is forced. 

    Args:
        SMC: An instance of an SMC sampler class

    """
    if SMC.lkernel == "asymptotic" and SMC.forward_kernel.accept_reject == False:
        warnings.warn("Warning: Accept-reject is false and therefore not a valid MCMC kernel. Setting accept-reject to true.")
        SMC.forward_kernel.accept_reject = True


def Set_MeanVar_Arrays(SMC):
    """
    Description:
        Sets the size of the mean and variance arrays which estimates are stored into. Models with constrained paramters will have these as additional dimensions

    Args:
        SMC: An instance of an SMC sampler class

    """
    # Hold etimated quantities and diagnostic metrics
    if hasattr(SMC.target, "constrained_dim"):
        SMC.mean_estimate = np.zeros([SMC.K + 1, SMC.target.constrained_dim])
        SMC.variance_estimate = np.zeros([SMC.K + 1, SMC.target.constrained_dim])
    else:
        SMC.mean_estimate = np.zeros([SMC.K + 1, SMC.target.dim])
        SMC.variance_estimate = np.zeros([SMC.K + 1, SMC.target.dim])