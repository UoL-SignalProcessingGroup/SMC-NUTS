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
- The BridgeStan Pythonic interface to Stan must be installed in order to sample from Stan models.
```
pip install bridgestan
```

## Using SMC-NUTS

An example is provided in the `examples` folder:
```
python run_experiments.py
```

**Example: An SMC sampler can be applied to a user-defined target density defined in Stan code as follows**

First we set the target distribution which is done via:
```
target = StanModel(model_name=model_name, model_path=str(model_path), data_path=str(model_data_path))
```
Here model_name is the name of the filename of a '.stan' file of interest which is located in the ```stan_models``` directory, model_path and model_data_path direct to the path of the model file and its associated data, respectively. 

Next we define some SMC options, such as the tempering stratergy, the initial proposal distribition (sample_proposal), and the proposal used to the initial momentum.
```
tempering = AdaptiveTempering(N=N, target=target, alpha=0.5)
sample_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim), seed=rng)
momentum_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim), seed=rng)
```

Next we define parameters for the proposal distribution itself, by setting the target, the momentum proposal to define the intial momentum of the proposal distribution, the step_size and the random seed
```
forward_kernel = NUTSProposal(
    target=target,
    momentum_proposal=momentum_proposal,
    step_size = step_size,
    rng=rng,
)
```
Finally we set the SMC sampler up for K iterations using N samples, using the parameters set above, and generate samples ,
```
forward_lkernel = ForwardLKernel(target=target, momentum_proposal=momentum_proposal)

SMC_NUTS= SMCSampler(
    K=K,
    N=N,
    target=target,
    forward_kernel=forward_kernel,
    sample_proposal=sample_proposal,
    lkernel=forward_lkernel,
    verbose=VERBOSE,
    rng=rng,
)

SMC_NUTS.sample()
```


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
