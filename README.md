# SMC-NUTS
### An SMC sampler with the No-U-Turn sampler proposal distribution 

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

## Contact

TO DO: ADD CONTACT

## Citation

We appreciate citations as they let us discover what people have been doing with the software. 

TO DO: ADD CITATION
