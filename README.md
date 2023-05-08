# LCAONet

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=grey)](https://pycqa.github.io/isort/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/nmdl-mizo/lcaonet/blob/main/LICENSE)

<img src="https://github.com/nmdl-mizo/lcaonet/blob/feat/image/image/model.png" width=400px alt="Model">

## Installation

### Requirements

- 3.7 <= [Python](https://www.python.org/) <= 3.10
- [NumPy](https://numpy.org/) == 1.*
- [SciPy](https://scipy.org/) == 1.*
- [SymPy](https://www.sympy.org/en/index.html) == 1.*
- [ASE](https://wiki.fysik.dtu.dk/ase/index.html) == 3.*
- [PyTorch](https://pytorch.org/) == **2.0.0**
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest)
- [PyTorch Scatter](https://pytorch-scatter.readthedocs.io/en/latest/)
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse)

***Note: Using a GPU is recommended.***

### Install from source

First, clone this repository.

```bash
git clone https://github.com/nmdl-mizo/lcaonet.git
```

It is possible to build a virtual environment using [conda](https://docs.conda.io/en/latest), [venv](https://docs.python.org/3/library/venv.html), or docker.

#### Using conda

You can create a new virtual environment with conda by running below commands:

```bash
conda create -n lcaonet python=3.10
conda activate lcaonet
```

Install dependencies in your environment:

```bash
conda install pytorch=2.0.0 -c pytorch
conda install pyg pytorch-scatter pytorch-sparse -c pyg
conda install numpy=1.* scipy=1.* sympy=1.* ase=3.* -c anaconda -c conda-forge
```

Install LCAONet:

```bash
cd lcaonet/conda
chmod +x build_conda.sh
./build_conda.sh
```

#### Using venv

You can create a new virtual environments with venv by running below commands:

```bash
python3 -m venv lcaonet-venv
source lcaonet-venv/bin/activate
```

Install dependencies in your environment:

```bash
cd lcaonet
pip install -r requirements.txt
```

Install LCAONet:

```bash
pip install .
```

#### Using docker

You can use the docker image of base environment from [here](https://hub.docker.com/r/ken2403/lcaonet-base).

## Usage

You can train LCAONet with custom data in the following three steps.

1. **Prepare data**

    First, prepare a list of `ase.Atoms` objects and a dict of physical property values to be labels. Then, convert them to `lcaonet.data.List2GraphDataset` object which inherits the `torch_geoemtric.data.Dataset` class.

    ```python
    from numpy import ndarray
    from torch import Tensor
    from ase import Atoms

    from lcaonet.data import List2GraphDataset

    # Prepare a list of Atoms objects
    data_list: list[Union[Atoms]] = ...
    # Prepare a dict of physical property values(Key: label name, Value: array of label values).
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
        n_per_orb: int = 1,
        cutoff: float | None = None,
        rbf_type: str | type[BaseRadialBasis] = "hydrogen",
        cutoff_net: str | type[BaseCutoff] | None = "polynomial",
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
    import torch
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

1. K. Nishio, K. Shibata, T. Mizoguchi. *LCAONet: Message passing with basis functions optimized by edge elemental species.* (2023) [Paper](https://arxiv.org/abs/)
