import setuptools

setup(
    name="smcnuts",
    version="0.1.0",
    description="A modular SMC sampler in Python.",
    author="Matthew Carter",
    author_email="m.j.carter2@liverpool.ac.uk",
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
