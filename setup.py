from setuptools import find_packages, setup

__version__ = "0.0.1"
URL = "https://github.com/nmdl-mizo/pyg_material"


install_requires = [
    "pyrootutils",
    "numpy",
    "sympy",
    "h5py",
    "ase",
    "pymatgen==2022.4.19" "torch>=1.7",
    "pyg-lib",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
    "torch-geometric",
    "hydra-core==1.2.0",
    "hydra-colorlog==1.2.0",
    "hydra-optuna-sweeper==1.2.0",
    "pytorch-lightning",
    "rich",
    "wandb",
]

test_requires = [
    "pytest",
    "pytest-cov",
]

dev_requires = test_requires + [
    "pre-commit",
]

setup(
    name="pyg_material",
    version=__version__,
    description="PyGMaterial - GNNs for Materials Informatics implemented in PyTorch Geometric",
    author="Kento Nishio",
    author_email="knishio@iis.u-tokyo.ac.jp",
    url=URL,
    keywords=[
        "deep-learning",
        "pytorch",
        "graph-neural-networks",
        "graph-convolutional-networks",
        "materials-informatics",
        "machine-learning-interatomic-potential",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": dev_requires,
    },
    packages=find_packages(),
    include_package_data=True,
)
