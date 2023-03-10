# LCAONet

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=grey)](https://pycqa.github.io/isort/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/nmdl-mizo/lcaonet#license)

## Installation

### Requirements

- 3.7 <= [Python](https://www.python.org/) <= 3.10
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [SymPy](https://www.sympy.org/en/index.html)
- [ASE](https://wiki.fysik.dtu.dk/ase/index.html) == 3.22.1
- [pymatgen](https://pymatgen.org/) == 2022.4.19
- [PyTorch](https://pytorch.org/) == 1.13.0
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest)

***Note: Using a GPU is recommended.***

### Prepare environment

You can create a new environment with [conda](https://docs.conda.io/en/latest/) by running below commands:

```bash
conda create -n lcaonet python=3.9
conda activate lcaonet
```

Install dependencies:

```bash
conda install -c conda-forge numpy scipy sympy ase=3.22.1 pymatgen=2022.4.19
conda install pytorch==1.13.0 -c pytorch
conda install pyg -c pyg
```

### Install from source

You can install the package from source by cloning the repository and running below commands:

```bash
git clone https://github.com/nmdl-mizo/lcaonet.git
cd lcaonet
pip install .
```

## Usage

You can train LCAONet with custom data in the following three steps.

1. **Prepare data**

    First, prepare a list of `pymatgen.core.Structure` or `ase.Atoms` objects and a dict of physical property values to be labels. Then, convert them to `lcaonet.data.List2GraphDataset` object that inherits the `torch_geoemtric.data.Dataset` class.

    ```python
    from numpy import ndarray
    from torch import Tensor
    from ase import Atoms
    from pymatgen.core import Structure

    from lcaonet.data import List2GraphDataset

    # Prepare a list of Structure or Atoms objects
    data_list: list[Union[Structure, Atoms]] = ...
    # Prepare a dict of physical property values(Key: label name, Value: label value).
    label_list: dict[str, list[float] | ndarray | Tensor] = ...

    # Convert to List2GraphDataset object
    dataset = List2GraphDataset(data_list, y_values=label_list, cutoff=5.0)
    ```

2. **Define model**

    Define LCAONet with any hyperparameters.

    ```python
    from lcaonet.nn.cutoff import BaseCutoff
    from lcaonet.model import LCAONet

    # Define LCAONet
    model = LCAONet(
        hidden_dim: int = 128,
        coeffs_dim: int = 128,
        conv_dim: int = 128,
        out_dim: int = 1,
        n_interaction: int = 3,
        cutoff: float | None = None,
        cutoff_net: type[BaseCutoff] | None = None,
        bohr_radius: float = 0.529,
        max_z: int = 36,
        max_orb: str | None = None,
        elec_to_node: bool = True,
        add_valence: bool = False,
        extend_orb: bool = False,
        is_extensive: bool = True,
        activation: str = "SiLU",
        weight_init: str | None = "glorotorthogonal",
    )
    ```

3. **Train model**

    Train with the interface of your choice (either plain PyTorch or PytorchLighting).

    ```python
    from torch_geometric.loader import DataLoader

    # Prepare DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        # Forward
        y_pred = model(batch)
        # Calculate loss
        loss = ...
        loss.backward()
        ...
    ```

## References

1. K. Nishio, K. Shibata, T. Mizoguchi. *LCAONet: Message passing with optimized basis functions for edge elemental species.* (2023) [Paper](https://arxiv.org/abs/)
