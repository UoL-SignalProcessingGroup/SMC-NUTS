# SMC-NUTS
### An SMC sampler with a No-U-Turn sampler proposal distribution 

This is a codebase used to generate results when using the No-U-Turn Sampler (NUTS) as a proposal distribution for a sequential Monte Carlo (SMC) sampler. This algorithm has been designed to take a [Stan](https://mc-stan.org/) model as input and use [BridgeStan](https://github.com/roualdes/bridgestan) to evaluate the both evaluations of the log posterior and the associated gradient. 


## Installing SMC-NUTS

To install SMC-NUTS, follow these steps:

```
git clone https://github.com/UoL-SignalProcessingGroup/SMC-NUTS

cd SMC-NUTS

python3 -m pip install -e .
```

**NOTE**:
- The BridgeStan Pythonic interface to Stan must be installed in order to sample from Stan models (requirement > Python3.8).
```
pip install bridgestan
```

## Using SMC-NUTS

An example is provided in the `examples` directory:
```
python run_experiments.py
```
This code will generate multiple results into the `experiments/output/model_name` for a model defined in `stan_models`. Results generated include: Mean estimates, variance estimates, effective sample size (ess), temperature (phi) schedule, and the acceptance probability. 

Results can then be plotted by 
```
python plot_experiments.py
```
Which will plot averaged results over all runs. 

## Defining an SMC sampler and its L-kernel strategy
An SMC sampler instance is created with the following call

```
SMC_NUTS= SMCSampler(
    K=K,
    N=N,
    target=target,
    forward_kernel=forward_kernel,
    sample_proposal=sample_proposal,
    tempering=tempering
    lkernel=forward_lkernel,
    verbose=VERBOSE,
    rng=rng,
)
```
Where `K` is the number of iterations the sampler is run over, `N` is the number of samples, target is a `StanModel` instance from Bridgestan, `forward_kernel` describes how samples are propagated inside the sampler, `tempering` describes the tempering strategy to be used, 'lkernel' tells the sampler what l-kernel to use and therefore which weight update needs to be implemented. 

At present there are three strategies define the L-kernel in the SMC sampler
- An Asymptopic L-kernel which uses accept-reject
- An L-kernel parameterised by the Forwards proposal 
- A Gaussian approximation to the Optimal-L-kernel

Please consult `experiments/run_experiments.py` to see examples of how this may be implemented in practice.

## Future updates
- Adaptive learning of both the mass matrix and step-size utilised by the sampler.

## Citation

We appreciate citations as they let us discover what people have been doing with the software. 

To cite SMC-NUTS in publications use:

Devlin, L., Carter, M., Green, P.L., Maskell, S. (2023). SMC-NUTS (1.0.0). https://github.com//UoL-SignalProcessingGroup/SMC-NUTS

Or use the following BibTeX entry:

```
@misc{pysmc,
  title = {SMC-NUTS (1.0.0)},
  author = { Devlin, L., Carter, M., Green, P.L., Maskell, S.},
  year = {2023},
  month = September,
  howpublished = {GitHub},
  url = {https://github.com//UoL-SignalProcessingGroup/SMC-NUTS}
}
```
