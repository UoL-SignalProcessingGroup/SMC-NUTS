import warnings
import numpy as np

"""
Here we set some functions which are used as prechecks before the SMC sampler is run.

"""


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