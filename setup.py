from setuptools import find_packages, setup

__version__ = "1.4.0"
URL = "https://github.com/nmdl-mizo/lcaonet"


install_requires = [
    "numpy==1.23.5",
    "scipy==1.10.1",
    "sympy==1.11.1",
    "ase==3.22.1",
    "torch==1.13.1",
    "torch_geometric",
    # "torch_scatter @ https://data.pyg.org/whl/torch-2.0.0+cu117.html",
    # "torch_sparse @ https://data.pyg.org/whl/torch-2.0.0+cu117.html",
]

test_requires = [
    "pytest",
    "pytest-cov",
]

dev_requires = test_requires + [
    "pre-commit",
    "black",
]

setup(
    name="lcaonet",
    version=__version__,
    description="LCAONet - MPNN including orbital interaction, physically motivatied by the LCAO method.",
    author="Kento Nishio",
    author_email="knishio@iis.u-tokyo.ac.jp",
    url=URL,
    keywords=[
        "deep-learning",
        "pytorch",
        "graph-neural-networks",
        "graph-convolutional-neural-networks",
        "materials-informatics",
        "machine-learning-interatomic-potential",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": dev_requires,
    },
    packages=find_packages(),
    include_package_data=True,
)
