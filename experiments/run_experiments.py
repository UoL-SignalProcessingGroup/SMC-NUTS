import json
from pathlib import Path

import autograd.numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal


from smcnuts.proposal.nuts import NUTSProposal #should be in samples
from smcnuts.smc_sampler import SMCSampler
from smcnuts.model.bridgestan import StanModel
from smcnuts.postprocessing.ess_tempering import estimate_from_tempered

sns.set_style("whitegrid")

"""
Script to run the SMC-sampler with different configurations for multiple Monte Carlo runs

The three configurations are:

i) An SMC-sampler using accept-reject
ii) An SMC-sampler parameterised by using the forwards proposal as the L-kernel
iii) An SMC-sampler parameterised by using a Gaussian approximation to the optimal-L kernel. 

Options:
N_MCMC_RUNS: Number of Monte Carlo runs
N: The number of iterations the sampler is ran for
k: THe number of samples used
Model_name: The name of the stan model being used, must be placed in '../stan_models/'
VERBOSE: Updates to terminal the current iteration

SMC configurations:
tempering : Set a tempering mechanism, default is None
sample_proposal : = Set an initial distribution of samples
step_size : step size for the numerical integration. Taken from '../stan_models/$Model_name$/config_model.json', otherwise defaults to 0.5
momentum_proposal : Set a distribution from which to sample a momentum value
accept_reject : Turn on the accept_reject mechanism
lkernel: Set L-kernel. Matching configurations above asymptoptic (i), forward_lkernel (ii), and gauss_lkernel (iii)
"""

#Number of Monte-Carlo runs
N_MCMC_RUNS = 25

# Sampler configurations
N = 100 #Number of samples
K = 15 #Number of iterations

# Specify model - CHANGE THIS TO CHANGE STAN MODEL
model_name = "arma"

VERBOSE = False

def main():

    output_dir = Path.joinpath(Path.cwd(), "output", model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stan model
    model_dir = Path.joinpath(Path.cwd().parent, "stan_models", model_name)
    model_path = Path.joinpath(model_dir, f"{model_name}.stan")
    if not model_path.exists():
        raise FileNotFoundError(f"Stan model {model_name} not found.")

    # Check for data
    model_data_path = Path.joinpath(model_dir, f"{model_name}.json")
    if not model_data_path.exists():
        model_data_path = None

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

    # Load model config
    model_config_path = Path.joinpath(model_dir, f"model_config.json")
    if model_config_path.exists():
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config = None
    
    # Load Step-size from model_config file
    if model_config is not None and "step_size" in model_config.keys():
        step_size = model_config["step_size"]
    else:
        step_size = 0.5

    # Load Stan model
    target = StanModel(model_name=model_name, model_path=str(model_path), data_path=str(model_data_path))

    print(f"Model: {model_name}")
    print(f"K: {K}")
    print(f"N: {N}")
    print(f"step_size: {step_size}")

    for i in range(N_MCMC_RUNS):
        print(f"\nMCMC Run {i + 1} of {N_MCMC_RUNS}")
        
        # Fix seed for particular iterations
        rng = np.random.RandomState(10 * (i + 1))

        # Initialize samplers
        sample_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim), seed=rng)
        momentum_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim), seed=rng)
        

        forward_kernel = NUTSProposal(
            target=target,
            momentum_proposal=momentum_proposal,
            step_size = step_size,
            rng=rng,
        )

        print("Sampling with Forward Proposal L Kernel")
        forward_lkernel = ForwardLKernel(target=target, momentum_proposal=momentum_proposal)
        fp_nuts_smcs = SMCSampler(
            K=K,
            N=N,
            target=target,
            forward_kernel=forward_kernel,
            sample_proposal=sample_proposal,
            lkernel=forward_lkernel,
            verbose=VERBOSE,
            rng=rng,
        )

        fp_nuts_smcs.sample()

        print(f"\nFinished sampling in {fp_nuts_smcs.run_time} seconds")

        fp_output_dir = Path.joinpath(output_dir, "forward_lkernel")
        fp_output_dir.mkdir(parents=True, exist_ok=True)

        # Save output to csv
        mean_estimate_path = Path.joinpath(fp_output_dir, f"mean_estimate_{i}.csv")
        np.savetxt(mean_estimate_path, fp_nuts_smcs.mean_estimate, delimiter=",")
        var_estimate_path = Path.joinpath(fp_output_dir, f"var_estimate_{i}.csv")
        np.savetxt(var_estimate_path, fp_nuts_smcs.variance_estimate, delimiter=",")
        ess_path = Path.joinpath(fp_output_dir, f"ess_{i}.csv")
        np.savetxt(ess_path, fp_nuts_smcs.ess, delimiter=",")
        phi_path = Path.joinpath(fp_output_dir, f"phi_{i}.csv")
        np.savetxt(phi_path, fp_nuts_smcs.phi, delimiter=",")
        acceptance_rate_path = Path.joinpath(fp_output_dir, f"acceptance_rate_{i}.csv")
        np.savetxt(acceptance_rate_path, fp_nuts_smcs.acceptance_rate, delimiter=",")

        print("Sampling with Gaussian Approximation L Kernel")
        gauss_lkernel = GaussianApproxLKernel(target=target, N=N)
        gauss_nuts_smcs = SMCSampler(
            K=K,
            N=N,
            target=target,
            forward_kernel=forward_kernel,
            sample_proposal=sample_proposal,
            lkernel=gauss_lkernel,
            verbose=VERBOSE,
            rng=rng,
        )

        gauss_nuts_smcs.sample()

        print(f"\nFinished sampling in {gauss_nuts_smcs.run_time} seconds")

        gauss_output_dir = Path.joinpath(output_dir, "gaussian_lkernel")
        gauss_output_dir.mkdir(parents=True, exist_ok=True)

        # Save output to csv
        mean_estimate_path = Path.joinpath(gauss_output_dir, f"mean_estimate_{i}.csv")
        np.savetxt(mean_estimate_path, gauss_nuts_smcs.mean_estimate, delimiter=",")
        var_estimate_path = Path.joinpath(gauss_output_dir, f"var_estimate_{i}.csv")
        np.savetxt(var_estimate_path, gauss_nuts_smcs.variance_estimate, delimiter=",")
        ess_path = Path.joinpath(gauss_output_dir, f"ess_{i}.csv")
        np.savetxt(ess_path, gauss_nuts_smcs.ess, delimiter=",")
        phi_path = Path.joinpath(gauss_output_dir, f"phi_{i}.csv")
        np.savetxt(phi_path, gauss_nuts_smcs.phi, delimiter=",")
        acceptance_rate_path = Path.joinpath(gauss_output_dir, f"acceptance_rate_{i}.csv")
        np.savetxt(acceptance_rate_path, gauss_nuts_smcs.acceptance_rate, delimiter=",")

        print("Sampling with Asymptotically Optimal L Kernel with Adaptive Tempering and Accept/Reject")
        forward_kernel = NUTSProposal(
            target=target,
            momentum_proposal=momentum_proposal,
            step_size = step_size,
            accept_reject=True,
            rng=rng,
        )
        tempered_nuts_smcs = SMCSampler(
            K=K,
            N=N,
            target=target,
            forward_kernel=forward_kernel,
            sample_proposal=sample_proposal,
            tempering=tempering,
            lkernel="asymptotic",
            verbose=VERBOSE,
            rng=rng,
        )

        tempered_nuts_smcs.sample(
            save_samples=True,
        )

        print(f"\nFinished sampling in {tempered_nuts_smcs.run_time} seconds")

        tempered_nuts_smcs = estimate_from_tempered(target, tempered_nuts_smcs)

        tempered_output_dir = Path.joinpath(output_dir, "asymptotic_lkernel")
        tempered_output_dir.mkdir(parents=True, exist_ok=True)

        # Save output to csv
        mean_estimate_path = Path.joinpath(tempered_output_dir, f"mean_estimate_{i}.csv")
        np.savetxt(mean_estimate_path, tempered_nuts_smcs.mean_estimate, delimiter=",")
        var_estimate_path = Path.joinpath(tempered_output_dir, f"var_estimate_{i}.csv")
        np.savetxt(var_estimate_path, tempered_nuts_smcs.variance_estimate, delimiter=",")
        ess_path = Path.joinpath(tempered_output_dir, f"ess_{i}.csv")
        np.savetxt(ess_path, tempered_nuts_smcs.ess, delimiter=",")
        phi_path = Path.joinpath(tempered_output_dir, f"phi_{i}.csv")
        np.savetxt(phi_path, tempered_nuts_smcs.phi, delimiter=",")
        acceptance_rate_path = Path.joinpath(tempered_output_dir, f"acceptance_rate_{i}.csv")
        np.savetxt(acceptance_rate_path, tempered_nuts_smcs.acceptance_rate, delimiter=",")


if __name__ == "__main__":
    main()
