from scipy.stats import multivariate_normal

def normal_distribution_lpdf(x,mean,covaiance):
    return multivariate_normal.logpdf(x, mean, covaiance)

    