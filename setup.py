import setuptools

setup(
    name="smcnuts",
    version="0.1.0",
    description="A modular SMC sampler in Python.",
    author="Lee Devlin, Matthew Carter, Paul Horridge, Peter L. Green, Simon Maskell",
    author_email="lee.devlin@liverpool.ac.uk, mcarter@liverpool.ac.uk, paul.horridge@liverpool.ac.uk, plgreen@liverpool.ac.uk, smaskell@liverpool.ac.uk",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19",
        "autograd>=1.6",
        "seaborn>=0.12",
        "matplotlib>=3.7",
        "scipy>=1.10",
        "tqdm>=4.65",
    ],
)
