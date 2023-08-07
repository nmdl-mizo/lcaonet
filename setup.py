import re

from setuptools import find_packages, setup

with open("lcaonet/__init__.py", encoding="utf-8") as fd:
    for line in fd.readlines():
        m = re.search('__version__ = "(.*)"', line)
        if m:
            version = m.group(1)
            break


install_requires = [
    "numpy==1.*",
    "scipy==1.*",
    "sympy==1.*",
    "ase==3.*",
    "torch==2.0",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
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
    version=version,
    description="LCAONet - MPNN including orbital interaction, physically motivatied by the LCAO method.",
    author="Kento Nishio",
    author_email="knishio@iis.u-tokyo.ac.jp",
    url="https://github.com/nmdl-mizo/lcaonet",
    keywords=[
        "deep-learning",
        "pytorch",
        "graph-neural-networks",
        "graph-convolutional-neural-networks",
        "materials-informatics",
        "machine-learning-interatomic-potential",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": dev_requires,
    },
    packages=find_packages(),
    include_package_data=True,
)
