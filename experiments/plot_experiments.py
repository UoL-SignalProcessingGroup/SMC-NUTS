from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

"""
Plotting script to visualise results from the smc sampler

Generates the following plots:
i) Raw mean estimates of the target distribution show the mean over N_MC_RUNS Monte carlo runs with standard deviation
ii) Recycled mean estimates of the target distribution show the mean over N_MC_RUNS Monte carlo runs with standard deviation
iii) Averaged Mean Square Estimates (MSE) for raw estimates compared to a 'gold-standard' obtained from a long running MCMC chain from Stan
iv) Averaged Mean Square Estimates (MSE) for Recycled estimates compared to a 'gold-standard' obtained from a long running MCMC chain from Stan

"""

#Number of Monte-Carlo runs
N_MC_RUNS = 25

# Specify model - CHANGE THIS TO CHANGE STAN MODEL
model_name = "arma"


def monte_carlo_moments_estimators(x, return_sd=True):
    """
    Calculate the Monte Carlo mean and variance of an estimator over a series of Monte Carlo runs.

    Input shape is (M, K, D) where
        M (x[0]) is the number of Monte Carlo runs,
        K (x[1]) is the number of estimates per Monte Carlo run, and
        D (x[2]) is the dimensionality of the model.
    """
    if len(x.shape) != 3:
        raise ValueError("Input must be a 3D tensor of shape (M, K, D) where M is the number of Monte Carlo runs, K is the number of estimates per Monte Carlo run, and D is the dimensionality of the model.")

    if x.shape[0] > 1:
        # Calculate mean of tensors
        mean = np.zeros([x.shape[1], x.shape[2]])
        for m in range(x.shape[0]):
            for k in range(x.shape[1]):
                mean[k] += x[m, k, :]
        mean /= x.shape[0]

        # Calculate variance of tensors
        variance = np.zeros([x.shape[1], x.shape[2]])
        for m in range(x.shape[0]):
            for k in range(x.shape[1]):
                variance[k] += (x[m, k, :] - mean[k])**2
        variance /= x.shape[0]

        if return_sd:
            variance = np.sqrt(variance)

        return mean, variance
    else:
        return x[0], np.zeros([x.shape[1], x.shape[2]])


def mse_mean_var(x, ground_truth, log_scale=False, return_sd=False):
    """
    Calculate the mean and variance of the mean squared error over several Monte Carlo runs

    Input shape is (M, K, D) where
        M is the number of Monte Carlo runs,
        K is the number of SMC iterations, and
        D is the dimensionality of the model.
    """

    mse_per_run_iter = np.zeros([x.shape[0], x.shape[1]])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mse_per_run_iter[i, j] = np.mean(np.square(x[i, j, :] - ground_truth))

    mse_mean = np.mean(mse_per_run_iter, axis=0)
    mse_var = np.var(mse_per_run_iter, axis=0)

    return mse_mean, mse_var


def main():

    output_dir = Path.joinpath(Path.cwd(), "output", model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stan model
    model_dir = Path.joinpath(Path.cwd().parent, "stan_models", model_name)

    # Load ground truth
    true_mean = []
    true_var = []
    ground_truth_path = Path.joinpath(model_dir, f"{model_name}.params")
    with open(ground_truth_path, "r") as f:
        for line in f:
            mean = line.split()[1]
            variance = line.split()[2]
            true_mean.append(float(mean))
            true_var.append(float(variance))

    asymptotic_mean_estimates = []
    forward_mean_estimates = []
    gaussian_mean_estimates = []

    for i in range(N_MC_RUNS):
        asymptotic_mean_estimate_file = Path(output_dir, f"asymptotic_lkernel/mean_estimate_{i}.csv")
        asymptotic_mean_estimates.append(np.loadtxt(asymptotic_mean_estimate_file, delimiter=","))
        forward_mean_estimate_file = Path(output_dir, f"forward_lkernel/mean_estimate_{i}.csv")
        forward_mean_estimates.append(np.loadtxt(forward_mean_estimate_file, delimiter=","))
        gaussian_mean_estimate_file = Path(output_dir, f"gaussian_lkernel/mean_estimate_{i}.csv")
        gaussian_mean_estimates.append(np.loadtxt(gaussian_mean_estimate_file, delimiter=","))

    asymptotic_mean_estimates = np.array(asymptotic_mean_estimates)
    forward_mean_estimates = np.array(forward_mean_estimates)
    gaussian_mean_estimates = np.array(gaussian_mean_estimates)

    asymptotic_mean_of_mean, asymptotic_sd_of_mean = monte_carlo_moments_estimators(asymptotic_mean_estimates, return_sd=True)
    fp_mean_of_mean, fp_sd_of_mean = monte_carlo_moments_estimators(forward_mean_estimates, return_sd=True)
    gauss_mean_of_mean, gauss_sd_of_mean = monte_carlo_moments_estimators(gaussian_mean_estimates, return_sd=True)

    asymptotic_mean_of_mean_mse, asymptotic_sd_of_mean_mse = mse_mean_var(asymptotic_mean_estimates, true_mean, return_sd=True)
    fp_mean_of_mean_mse, fp_sd_of_mean_mse = mse_mean_var(forward_mean_estimates, true_mean, return_sd=True)
    gauss_mean_of_mean_mse, gauss_sd_of_mean_mse = mse_mean_var(gaussian_mean_estimates, true_mean, return_sd=True)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    for i in range(len(true_mean)):
        axs[0].axhline(y=true_mean[i], linestyle="--", color="g")
        axs[1].axhline(y=true_mean[i], linestyle="--", color="g")
        axs[2].axhline(y=true_mean[i], linestyle="--", color="g")
        axs[0].plot(asymptotic_mean_of_mean[:, i], "k", label="Accept/Reject with tempering")
        axs[0].fill_between(range(len(asymptotic_mean_of_mean)), asymptotic_mean_of_mean[:, i] - asymptotic_sd_of_mean[:, i], asymptotic_mean_of_mean[:, i] + asymptotic_sd_of_mean[:, i], color='orange', alpha=0.2)
        axs[0].set_title("Accept/Reject with tempering")
        axs[1].plot(fp_mean_of_mean[:, i], "b", label="Forwards proposal")
        axs[1].fill_between(range(len(asymptotic_mean_of_mean)), fp_mean_of_mean[:, i] - fp_sd_of_mean[:, i], fp_mean_of_mean[:, i] + fp_sd_of_mean[:, i], color='orange', alpha=0.2)
        axs[1].set_title("Forwards proposal")
        axs[2].plot(gauss_mean_of_mean[:, i], "r", label="Gaussian approximation")
        axs[2].fill_between(range(len(asymptotic_mean_of_mean)), gauss_mean_of_mean[:, i] - gauss_sd_of_mean[:, i], gauss_mean_of_mean[:, i] + gauss_sd_of_mean[:, i], color='orange', alpha=0.2)
        axs[2].set_title("Gaussian approximation")
    for ax in axs.flat:
        ax.set(xlabel="Iteration", ylabel=r"E[$x$]")
    plt.tight_layout()
    plt.savefig(f"{model_name}_mean.png")

    plt.figure(figsize=(10, 5))
    plt.plot(asymptotic_mean_of_mean_mse, "k", label="Accept/Reject with tempering")
    plt.plot(fp_mean_of_mean_mse, "b", label="Forwards proposal")
    plt.plot(gauss_mean_of_mean_mse, "r", label="Gaussian approximation")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{model_name}_mse.png")

    asymptotic_mean_estimates = []
    forward_mean_estimates = []
    gaussian_mean_estimates = []

    for i in range(N_MC_RUNS):
        asymptotic_mean_estimate_file = Path(output_dir, f"asymptotic_lkernel/recycled_mean_estimate_{i}.csv")
        asymptotic_mean_estimates.append(np.loadtxt(asymptotic_mean_estimate_file, delimiter=","))
        forward_mean_estimate_file = Path(output_dir, f"forward_lkernel/recycled_mean_estimate_{i}.csv")
        forward_mean_estimates.append(np.loadtxt(forward_mean_estimate_file, delimiter=","))
        gaussian_mean_estimate_file = Path(output_dir, f"gaussian_lkernel/recycled_mean_estimate_{i}.csv")
        gaussian_mean_estimates.append(np.loadtxt(gaussian_mean_estimate_file, delimiter=","))

    asymptotic_mean_estimates = np.array(asymptotic_mean_estimates)
    forward_mean_estimates = np.array(forward_mean_estimates)
    gaussian_mean_estimates = np.array(gaussian_mean_estimates)

    asymptotic_mean_of_mean, asymptotic_sd_of_mean = monte_carlo_moments_estimators(asymptotic_mean_estimates, return_sd=True)
    fp_mean_of_mean, fp_sd_of_mean = monte_carlo_moments_estimators(forward_mean_estimates, return_sd=True)
    gauss_mean_of_mean, gauss_sd_of_mean = monte_carlo_moments_estimators(gaussian_mean_estimates, return_sd=True)

    asymptotic_mean_of_mean_mse, asymptotic_sd_of_mean_mse = mse_mean_var(asymptotic_mean_estimates, true_mean, return_sd=True)
    fp_mean_of_mean_mse, fp_sd_of_mean_mse = mse_mean_var(forward_mean_estimates, true_mean, return_sd=True)
    gauss_mean_of_mean_mse, gauss_sd_of_mean_mse = mse_mean_var(gaussian_mean_estimates, true_mean, return_sd=True)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    for i in range(len(true_mean)):
        axs[0].axhline(y=true_mean[i], linestyle="--", color="g")
        axs[1].axhline(y=true_mean[i], linestyle="--", color="g")
        axs[2].axhline(y=true_mean[i], linestyle="--", color="g")
        axs[0].plot(asymptotic_mean_of_mean[:, i], "k", label="Accept/Reject with tempering")
        axs[0].fill_between(range(len(asymptotic_mean_of_mean)), asymptotic_mean_of_mean[:, i] - asymptotic_sd_of_mean[:, i], asymptotic_mean_of_mean[:, i] + asymptotic_sd_of_mean[:, i], color='orange',alpha=0.2)
        axs[0].set_title("Accept/Reject with tempering")
        axs[1].plot(fp_mean_of_mean[:, i], "b", label="Forwards proposal")
        axs[1].fill_between(range(len(asymptotic_mean_of_mean)), fp_mean_of_mean[:, i] - fp_sd_of_mean[:, i], fp_mean_of_mean[:, i] + fp_sd_of_mean[:, i], color='orange', alpha=0.2)
        axs[1].set_title("Forwards proposal")
        axs[2].plot(gauss_mean_of_mean[:, i], "r", label="Gaussian approximation")
        axs[2].fill_between(range(len(asymptotic_mean_of_mean)), gauss_mean_of_mean[:, i] - gauss_sd_of_mean[:, i], gauss_mean_of_mean[:, i] + gauss_sd_of_mean[:, i], color='orange',alpha=0.2)
        axs[2].set_title("Gaussian approximation")
    for ax in axs.flat:
        ax.set(xlabel="Iteration", ylabel=r"E[$x$]")
    plt.tight_layout()
    plt.savefig(f"{model_name}_recycled_mean.png")

    plt.figure(figsize=(10, 5))
    plt.plot(asymptotic_mean_of_mean_mse, "k", label="Accept/Reject with tempering")
    plt.plot(fp_mean_of_mean_mse, "b", label="Forwards proposal")
    plt.plot(gauss_mean_of_mean_mse, "r", label="Gaussian approximation")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{model_name}_recycled_mse.png")


if __name__ == "__main__":
  main()
