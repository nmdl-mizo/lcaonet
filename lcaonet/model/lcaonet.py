from __future__ import annotations

import math
from collections.abc import Callable
from math import pi

import numpy as np
import sympy as sym
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import quad
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter

from lcaonet.data import DataKeys
from lcaonet.model.base import BaseMPNN
from lcaonet.nn import Dense
from lcaonet.nn.cutoff import BaseCutoff
from lcaonet.nn.post import PostProcess
from lcaonet.utils.resolve import activation_resolver, init_resolver

# 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d
ELEC_TABLE = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # dummy
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # H
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # He
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Li
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Be
        [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
        [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C
        [2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N
        [2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O
        [2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # F
        [2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ne
        [2, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Na
        [2, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mg
        [2, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Al
        [2, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Si
        [2, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # P
        [2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # S
        [2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cl
        [2, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ar (18)
        [2, 2, 6, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # K
        [2, 2, 6, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ca
        [2, 2, 6, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sc
        [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ti
        [2, 2, 6, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # V
        [2, 2, 6, 2, 6, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cr
        [2, 2, 6, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mn
        [2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Fe
        [2, 2, 6, 2, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Co
        [2, 2, 6, 2, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ni
        [2, 2, 6, 2, 6, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cu
        [2, 2, 6, 2, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zn
        [2, 2, 6, 2, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ga
        [2, 2, 6, 2, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ge
        [2, 2, 6, 2, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # As
        [2, 2, 6, 2, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Se
        [2, 2, 6, 2, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Br
        [2, 2, 6, 2, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Kr (36)
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sr
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Y
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],  # Zr
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],  # Nb
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0],  # Mo
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0],  # Tc
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0],  # Ru
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0],  # Rh
        [2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Pd
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Ag
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Cd
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0],  # In
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0],  # Sn
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0],  # Sb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0],  # Te
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0],  # I
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0],  # Xe (54)
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0],  # Cs
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0],  # Ba
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 1, 0, 0, 0, 0],  # La
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 1, 1, 0, 0, 0, 0],  # Ce
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 3, 0, 0, 0, 0, 0],  # Pr
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 4, 0, 0, 0, 0, 0],  # Nd
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0],  # Pm
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 6, 0, 0, 0, 0, 0],  # Sm
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 0, 0, 0, 0, 0],  # Eu
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 1, 0, 0, 0, 0],  # Gd
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 9, 0, 0, 0, 0, 0],  # Tb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0],  # Dy
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11, 0, 0, 0, 0, 0],  # Ho
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12, 0, 0, 0, 0, 0],  # Er
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13, 0, 0, 0, 0, 0],  # Tm
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 0, 0, 0, 0, 0],  # Yb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 1, 0, 0, 0, 0],  # Lu (71)
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 2, 0, 0, 0, 0],  # Hf
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 3, 0, 0, 0, 0],  # Ta
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 4, 0, 0, 0, 0],  # W
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 5, 0, 0, 0, 0],  # Re
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 6, 0, 0, 0, 0],  # Os
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 7, 0, 0, 0, 0],  # Ir
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 9, 0, 0, 0, 0],  # Pt
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 0, 0, 0],  # Au
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 0, 0, 0],  # Hg
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 0, 0, 0],  # Tl
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 0, 0, 0],  # Pb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 0, 0, 0],  # Bi
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 0, 0, 0],  # Po
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 0, 0, 0],  # At
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 0, 0, 0],  # Rn (86)
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 1, 0, 0],  # Fr
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 0],  # Ra
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 1],  # Ac
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 2],  # Th
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 2, 1],  # Pa
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 3, 1],  # U
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 4, 1],  # Np
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 0],  # Pu
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 0],  # Am
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 1],  # Cm (96)
    ]
)
MAX_IDX = [3, 3, 7, 3, 7, 3, 11, 7, 3, 11, 7, 3, 15, 11, 7, 3, 15, 11]

# 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d
VALENCE_TABLE = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # dummy
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # H
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # He
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Li
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Be
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # F
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ne
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Na
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mg
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Al
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Si
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # P
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # S
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cl
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ar (18)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # K
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ca
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sc
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ti
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # V
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cr
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mn
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Fe
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Co
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ni
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cu
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zn
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ga
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ge
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # As
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Se
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Br
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Kr (36)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rb
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sr
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Y
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Zr
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Nb
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Mo
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Tc
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ru
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Rh
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Pd
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ag
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Cd
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # In
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Sn
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Sb
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Te
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # I
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Xe (54)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Cs
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Ba
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # La
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Ce
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Pr
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Nd
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Pm
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Sm
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Eu
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Gd
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Tb
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Dy
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Ho
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Er
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Tm
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Yb
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Lu (71)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Hf
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Ta
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # W
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Re
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Os
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Ir
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Pt
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Au
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Hg
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Tl
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Pb
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Bi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Po
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # At
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Rn (86)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Fr
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Ra
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # Ac
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # Th
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # Pa
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # U
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # Np
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # Pu
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # Am
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # Cm (96)
    ]
)
N_ORB_BIG = 37

# fmt: off
NL_LIST_BIG: list[tuple[int, int]] = [
    (1, 0),                             # 1s
    (2, 0),                             # 2s
    (2, 1), (2, 1),                     # 2p
    (3, 0),                             # 3s
    (3, 1), (3, 1),                     # 3p
    (4, 0),                             # 4s
    (3, 2), (3, 2), (3, 2),             # 3d
    (4, 1), (4, 1),                     # 4p
    (5, 0),                             # 5s
    (4, 2), (4, 2), (4, 2),             # 4d
    (5, 1), (5, 1),                     # 5p
    (6, 0),                             # 6s
    (4, 3), (4, 3), (4, 3), (4, 3),     # 4f
    (5, 2), (5, 2), (5, 2),             # 5d
    (6, 1), (6, 1),                     # 6p
    (7, 0),                             # 7s
    (5, 3), (5, 3), (5, 3), (5, 3),     # 5f
    (6, 2), (6, 2), (6, 2),             # 6d
]
# fmt: on


def modify_elec_table(elec: Tensor = ELEC_TABLE) -> Tensor:
    # 1s,
    # 2s,
    # 2p, 2p,
    # 3s,
    # 3p, 3p,
    # 4s,
    # 3d, 3d, 3d,
    # 4p, 4p,
    # 5s,
    # 4d, 4d, 4d,
    # 5p, 5p,
    # 6s,
    # 4f, 4f, 4f, 4f,
    # 5d, 5d, 5d,
    # 6p, 6p,
    # 7s
    # 5f, 5f, 5f, 5f,
    # 6d, 6d, 6d,

    def modify_one(one_elec: Tensor) -> Tensor:
        out_elec = torch.zeros((N_ORB_BIG,), dtype=torch.long)
        j = 0
        for i in range(one_elec.size(0)):
            # s
            if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                out_elec[j] = one_elec[i]
                j += 1
            # p
            if np.isin([2, 4, 7, 10, 14], i).any():
                for _ in range(2):
                    out_elec[j] = one_elec[i]
                    j += 1
            # d
            if np.isin([6, 9, 13, 17], i).any():
                for _ in range(3):
                    out_elec[j] = one_elec[i]
                    j += 1
            # f
            if np.isin([12, 16], i).any():
                for _ in range(4):
                    out_elec[j] = one_elec[i]
                    j += 1
        return out_elec

    out_elec = torch.zeros((len(elec), N_ORB_BIG), dtype=torch.long)
    for i in range(len(elec)):
        out_elec[i] = modify_one(elec[i])

    return out_elec


def modify_max_ind(max_idx: list[int] = MAX_IDX) -> Tensor:
    out_max_ind = torch.zeros((N_ORB_BIG,), dtype=torch.long)
    j = 0
    for i in range(len(max_idx)):
        # s
        if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
            out_max_ind[j] = max_idx[i]
            j += 1
        # p
        if np.isin([2, 4, 7, 10, 14], i).any():
            for _ in range(2):
                out_max_ind[j] = max_idx[i]
                j += 1
        # d
        if np.isin([6, 9, 13, 17], i).any():
            for _ in range(3):
                out_max_ind[j] = max_idx[i]
                j += 1
        # f
        if np.isin([12, 16], i).any():
            for _ in range(4):
                out_max_ind[j] = max_idx[i]
                j += 1
    return out_max_ind


ELEC_TABLE_BIG = modify_elec_table()
VALENCE_TABLE_BIG = modify_elec_table(VALENCE_TABLE)
MAX_IDX_BIG = modify_max_ind()


def get_max_nl_index_byz(max_z: int) -> int:
    if max_z <= 2:
        return 0
    if max_z <= 4:
        return 1
    if max_z <= 10:
        return 3
    if max_z <= 12:
        return 4
    if max_z <= 18:
        return 6
    if max_z <= 20:
        return 7
    if max_z <= 30:
        return 10
    if max_z <= 36:
        return 12
    if max_z <= 38:
        return 13
    if max_z <= 48:
        return 16
    if max_z <= 54:
        return 18
    if max_z <= 56:
        return 19
    if max_z <= 80:
        return 26
    if max_z <= 86:
        return 28
    if max_z <= 88:
        return 29
    if max_z <= 96:
        return 36
    raise ValueError(f"max_z={max_z} is too large")


def get_max_nl_index_byorb(max_orb: str) -> int:
    if max_orb == "1s":
        return 0
    if max_orb == "2s":
        return 1
    if max_orb == "2p":
        return 3
    if max_orb == "3s":
        return 4
    if max_orb == "3p":
        return 6
    if max_orb == "4s":
        return 7
    if max_orb == "3d":
        return 10
    if max_orb == "4p":
        return 12
    if max_orb == "5s":
        return 13
    if max_orb == "4d":
        return 16
    if max_orb == "5p":
        return 18
    if max_orb == "6s":
        return 19
    if max_orb == "4f":
        return 23
    if max_orb == "5d":
        return 26
    if max_orb == "6p":
        return 28
    if max_orb == "7s":
        return 29
    if max_orb == "5f":
        return 33
    if max_orb == "6d":
        return 36
    raise ValueError(f"max_orb={max_orb} is not supported")


def get_elec_table(max_z: int, max_idx: int) -> Tensor:
    return ELEC_TABLE_BIG[: max_z + 1, : max_idx + 1]


def get_valence_table(max_z: int, max_idx: int) -> Tensor:
    return VALENCE_TABLE_BIG[: max_z + 1, : max_idx + 1]


def get_max_idx(max_z: int) -> Tensor:
    return MAX_IDX_BIG[: max_z + 1]


def get_nl_list(max_idx: int) -> list[tuple[int, int]]:
    return NL_LIST_BIG[: max_idx + 1]


class RadialOrbitalBasis(nn.Module):
    """The layer that expand the interatomic distance with the radial
    wavefunctions of hydrogen."""

    def __init__(
        self,
        cutoff: float | None = None,
        bohr_radius: float = 0.529,
        max_z: int = 36,
        max_orb: str | None = None,
    ):
        """
        Args:
            cutoff (float | None, optional): the cutoff radius. Defaults to `None`.
            bohr_radius (float | None, optional): the bohr radius. Defaults to `0.529`.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
            max_orb (str | None, optional): the maximum orbital name like "2p". Defaults to `None`.
        """
        super().__init__()
        # get elec table
        if max_orb is None:
            max_idx = get_max_nl_index_byz(max_z)
        else:
            max_idx = get_max_nl_index_byorb(max_orb)
        self.n_orb = max_idx + 1
        self.n_l_list = get_nl_list(max_idx)
        self.cutoff = cutoff
        self.bohr_radius = bohr_radius

        self.basis_func = []
        self.stand_coeff = []
        for n, l in self.n_l_list:
            r_nl = self._get_r_nl(n, l, self.bohr_radius)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                self.stand_coeff.append(self._get_standardized_coeff(r_nl).requires_grad_(True))

    def _get_r_nl(self, nq: int, lq: int, r0: float = 0.529) -> Callable[[Tensor | float], Tensor | float]:
        """Get RadialOrbitalBasis functions with the associated Laguerre
        polynomial.

        Args:
            nq (int): principal quantum number.
            lq (int): azimuthal quantum number.
            r0 (float): bohr radius. Defaults to `0.529`.

        Returns:
            r_nl (Callable[[Tensor | float], Tensor | float]): Orbital Basis function.
        """
        x = sym.Symbol("x", real=True)
        # modify for n, l parameter
        # ref: https://zenn.dev/shittoku_xxx/articles/13afd6fdfac44e
        assoc_lag_coeff = sym.lambdify(
            [x],
            sym.simplify(
                sym.assoc_laguerre(nq - lq - 1, 2 * lq + 1, x) * sym.factorial(nq + lq) * (-1) ** (2 * lq + 1)
            ),
        )
        if self.cutoff is not None:
            stand_coeff = -2.0 / nq / r0
        else:
            # standardize in all space
            stand_coeff = -math.sqrt(
                (2.0 / nq / r0) ** 3 * math.factorial(nq - lq - 1) / 2.0 / nq / math.factorial(nq + lq) ** 3
            )

        def r_nl(r: Tensor | float) -> Tensor | float:
            zeta = 2.0 / nq / r0 * r

            if isinstance(r, float):
                return stand_coeff * assoc_lag_coeff(zeta) * zeta**lq * math.exp(-zeta / 2.0)

            return stand_coeff * assoc_lag_coeff(zeta) * torch.pow(zeta, lq) * torch.exp(-zeta / 2.0)  # type: ignore

        return r_nl

    def _get_standardized_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        """If a cutoff radius is specified, the standardization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float], Tensor | float]): radial wave function.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Standardization coefficient such that the probability of existence
                within the cutoff sphere is 1.
        """
        if self.cutoff is None:
            raise ValueError("cutoff is None")
        cutoff = self.cutoff

        with torch.no_grad():

            def interad_func(r):
                return (r * func(r)) ** 2

            inte = quad(interad_func, 0.0, cutoff)
            return 1 / (torch.sqrt(torch.tensor([inte[0]])) + 1e-12)

    def forward(self, dist: Tensor) -> Tensor:
        """Forward calculation of RadialOrbitalBasis.

        Args:
            dist (torch.Tensor): the interatomic distance with (n_edge) shape.

        Returns:
            rbf (torch.Tensor): the expanded distance with (n_edge, n_orb) shape.
        """
        if self.cutoff is not None:
            device = dist.device
            rbf = torch.stack([f(dist) * sc.to(device) for f, sc in zip(self.basis_func, self.stand_coeff)], dim=1)
        else:
            rbf = torch.stack([f(dist) for f in self.basis_func], dim=1)  # type: ignore
        return rbf


class SphericalHarmonicsBasis(nn.Module):
    """The layer that expand interatomic distance and angles with radial
    wavefunctions of hydrogen and spherical harmonics functions."""

    def __init__(
        self,
        cutoff: float | None = None,
        bohr_radius: float = 0.529,
        max_z: int = 36,
        max_orb: str | None = None,
    ):
        """
        Args:
            cutoff (float | None, optional): the cutoff radius. Defaults to `None`.
            bohr_radius (float | None, optional): the bohr radius. Defaults to `0.529`.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
            max_orb (str | None, optional): the maximum orbital name like "2p". Defaults to `None`.
        """
        super().__init__()
        self.radial_basis = RadialOrbitalBasis(cutoff, bohr_radius, max_z=max_z, max_orb=max_orb)

        # make spherical basis functions
        self.sph_funcs = self._calculate_symbolic_sh_funcs()

    @staticmethod
    def _y00(theta: Tensor, phi: Tensor) -> Tensor:
        r"""
        Spherical Harmonics with `l=m=0`.
        ..math::
            Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

        Args:
            theta: the azimuthal angle.
            phi: the polar angle.

        Returns:
            `Y_0^0`: the spherical harmonics with `l=m=0`.
        """
        dtype = theta.dtype
        return (0.5 * torch.ones_like(theta) * math.sqrt(1.0 / pi)).to(dtype)

    def _calculate_symbolic_sh_funcs(self) -> list:
        """Calculate symbolic spherical harmonics functions.

        Returns:
            funcs (list[Callable]): the list of spherical harmonics functions.
        """
        funcs = []
        theta, phi = sym.symbols("theta phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "conjugate": torch.conj, "sqrt": torch.sqrt, "exp": torch.exp}
        for nl in self.radial_basis.n_l_list:
            # !! only m=zero is used
            m_list = [0]
            for m in m_list:
                if nl[1] == 0:
                    funcs.append(SphericalHarmonicsBasis._y00)
                else:
                    func = sym.functions.special.spherical_harmonics.Znm(nl[1], m, theta, phi).expand(func=True)
                    func = sym.simplify(func).evalf()
                    funcs.append(sym.lambdify([theta, phi], func, modules))
        self.orig_funcs = funcs

        return funcs

    def forward(self, dist: Tensor, angle: Tensor, edge_idx_kj: torch.LongTensor) -> Tensor:
        """Forward calculation of SphericalHarmonicsBasis.

        Args:
            dist (torch.Tensor): the interatomic distance with (n_edge) shape.
            angle (torch.Tensor): the angles of triplets with (n_triplets) shape.
            edge_idx_kj (torch.LongTensor): the edge index from atom k to j with (n_triplets) shape.

        Returns:
            torch.Tensor: the expanded distance and angles of (n_triplets, n_orb) shape.
        """
        # (n_edge, n_orb)
        rob = self.radial_basis(dist)
        # (n_triplets, n_orb)
        shb = torch.stack([f(angle, None) for f in self.sph_funcs], dim=1)

        # (n_triplets, n_orb)
        return rob[edge_idx_kj] * shb


class EmbedZ(nn.Module):
    """The layer that embeds atomic numbers into latent vectors."""

    def __init__(self, embed_dim: int, max_z: int = 36):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.z_embed = nn.Embedding(max_z + 1, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.z_embed.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, z: Tensor) -> Tensor:
        """Forward calculation of EmbedZ.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.

        Returns:
            z_embed (torch.Tensor): the embedding vectors with (n_node, embed_dim) shape.
        """
        return self.z_embed(z)


class EmbedElec(nn.Module):
    """The layer that embeds electron numbers into latent vectors.

    If `extend_orb=False`, then if the number of electrons in the ground
    state is zero, the orbital is a zero vector embedding.
    """

    def __init__(self, embed_dim: int, max_z: int = 36, max_orb: str | None = None, extend_orb: bool = False):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
            max_orb (str | None, optional): the maximum orbital name like "2p". Defaults to `None`.
            extend_orb (bool, optional): Whether to use an extended basis. Defaults to `False`.
        """
        super().__init__()
        # get elec table
        if max_orb is None:
            max_idx = get_max_nl_index_byz(max_z)
        else:
            max_idx = get_max_nl_index_byorb(max_orb)
        self.register_buffer("elec", get_elec_table(max_z, max_idx))
        self.n_orb = max_idx + 1

        self.embed_dim = embed_dim
        self.extend_orb = extend_orb
        self.e_embeds = nn.ModuleList(
            [nn.Embedding(m, embed_dim, padding_idx=None if extend_orb else 0) for m in MAX_IDX_BIG[: self.n_orb]]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for ee in self.e_embeds:
            ee.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
            # set padding_idx to zero
            if not self.extend_orb:
                ee._fill_padding_idx_with_zero()

    def forward(self, z: Tensor, z_embed: Tensor) -> Tensor:
        """Forward calculation of EmbedElec.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.
            z_embed (torch.Tensor): the embedding of atomic numbers with (n_node, embed_dim) shape.

        Returns:
            e_embed (torch.Tensor): the embedding of electron numbers with (n_node, n_orb, embed_dim) shape.
        """
        # (n_node, n_orb)
        elec = self.elec[z]  # type: ignore
        # (n_orb, n_node)
        elec = torch.transpose(elec, 0, 1)
        # (n_orb, n_node, embed_dim)
        e_embed = torch.stack([ce(elec[i]) for i, ce in enumerate(self.e_embeds)], dim=0)
        # (n_node, n_orb, embed_dim)
        e_embed = torch.transpose(e_embed, 0, 1)

        # (n_node) -> (n_node, 1, embed_dim)
        z_embed = z_embed.unsqueeze(1)
        # inject atomic information to e_embed vectors
        e_embed = e_embed + e_embed * z_embed

        return e_embed


class ValenceMask(nn.Module):
    """The layer that generates valence orbital mask.

    Only the coefficients for valence orbitals are set to 1, and the
    coefficients for all other orbitals (including inner-shell orbitals)
    are set to 0.
    """

    def __init__(self, embed_dim: int, max_z: int = 36, max_orb: str | None = None):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
            max_orb (str | None, optional): the maximum orbital name like "2p". Defaults to `None`.
        """
        super().__init__()
        # get valence table
        if max_orb is None:
            max_idx = get_max_nl_index_byz(max_z)
        else:
            max_idx = get_max_nl_index_byorb(max_orb)
        self.register_buffer("valence", get_valence_table(max_z, max_idx))
        self.n_orb = max_idx + 1

        self.embed_dim = embed_dim

    def forward(self, z: Tensor) -> Tensor:
        """Forward calculation of ValenceMask.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.

        Returns:
            valence_mask (torch.Tensor): valence orbital mask with (n_node, n_orb, embed_dim) shape.
        """
        valence_mask = self.valence[z]  # type: ignore
        return valence_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)


class EmbedNode(nn.Module):
    """The layer that embeds atomic numbers and electron numbers into node
    embedding vectors."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        use_elec: bool,
        e_dim: int | None = None,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node vector.
            z_dim (int): the dimension of atomic number embedding.
            use_elec (bool): whether to use electron number embedding.
            e_dim (int | None): the dimension of electron number embedding.
            activation (nn.Module, optional): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None, optional): the weight initialization function.
                Defaults to `None`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_elec = use_elec
        if use_elec:
            assert e_dim is not None
            self.e_dim = e_dim
        else:
            self.e_dim = 0

        self.z_e_lin = nn.Sequential(
            activation,
            Dense(z_dim + self.e_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim, False, weight_init),
        )

    def forward(self, z_embed: Tensor, e_embed: Tensor | None = None) -> Tensor:
        """Forward calculation of EmbedNode.

        Args:
            z_embed (torch.Tensor): the embedding of atomic numbers with (n_node, z_dim) shape.
            e_embed (torch.Tensor | None): the embedding of electron numbers with (n_node, n_orb, e_dim) shape.

        Returns:
            torch.Tensor: node embedding vectors with (n_node, hidden_dim) shape.
        """
        if self.use_elec:
            if e_embed is None:
                raise ValueError("e_embed must be set when use_elec is True.")
            z_e_embed = torch.cat([z_embed, e_embed.sum(1)], dim=-1)
        else:
            z_e_embed = z_embed
        return self.z_e_lin(z_e_embed)


class LCAOConv(nn.Module):
    """The layer that performs LCAO convolution."""

    def __init__(
        self,
        hidden_dim: int,
        coeffs_dim: int,
        conv_dim: int,
        add_valence: bool = False,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node vector.
            coeffs_dim (int): the dimension of coefficient vectors.
            conv_dim (int): the dimension of embedding vectors at convolution.
            add_valence (bool, optional): whether to add the effect of valence orbitals. Defaults to `False`.
            activation (nn.Module, optional): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None, optional): the weight initialization function.
                Defaults to `None`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coeffs_dim = coeffs_dim
        self.conv_dim = conv_dim
        self.add_valence = add_valence

        self.node_before_lin = Dense(hidden_dim, 2 * conv_dim, True, weight_init)

        # No bias is used to keep 0 coefficient vectors at 0
        out_dim = 3 * conv_dim if add_valence else 2 * conv_dim
        self.coeffs_before_lin = nn.Sequential(
            activation,
            Dense(coeffs_dim, conv_dim, False, weight_init),
            activation,
            Dense(conv_dim, out_dim, False, weight_init),
        )

        three_out_dim = 2 * conv_dim if add_valence else conv_dim
        self.three_lin = nn.Sequential(
            activation,
            Dense(conv_dim, three_out_dim, True, weight_init),
        )

        self.node_lin = nn.Sequential(
            activation,
            Dense(conv_dim + conv_dim, conv_dim, True, weight_init),
            activation,
            Dense(conv_dim, conv_dim, True, weight_init),
        )
        self.node_after_lin = Dense(conv_dim, hidden_dim, False, weight_init)

    def forward(
        self,
        x: Tensor,
        cji: Tensor,
        valence_mask: Tensor | None,
        cutoff_w: Tensor | None,
        robs: Tensor,
        shbs: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        tri_idx_k: Tensor,
        edge_idx_kj: torch.LongTensor,
        edge_idx_ji: torch.LongTensor,
    ) -> Tensor:
        """Forward calculation of LCAOConv.

        Args:
            x (torch.Tensor): node embedding vectors with (n_node, hidden_dim) shape.
            cji (torch.Tensor): coefficient vectors with (n_edge, n_orb, coeffs_dim) shape.
            valence_mask (torch.Tensor | None): valence orbital mask with (n_node, n_orb, conv_dim) shape.
            cutoff_w (torch.Tensor | None): cutoff weight with (n_edge) shape.
            robs (torch.Tensor): the radial orbital basis with (n_edge, n_orb) shape.
            shbs (torch.Tensor): the spherical harmonics basis with (n_triplets, n_orb) shape.
            idx_i (torch.Tensor): the indices of the first node of each edge with (n_edge) shape.
            idx_j (torch.Tensor): the indices of the second node of each edge with (n_edge) shape.
            tri_idx_k (torch.Tensor): the indices of the third node of each triplet with (n_triplets) shape.
            edge_idx_kj (torch.LongTensor): the edge index from atom k to j with (n_triplets) shape.
            edge_idx_ji (torch.LongTensor): the edge index from atom j to i with (n_triplets) shape.

        Returns:
            torch.Tensor: updated node embedding vectors with (n_node, hidden_dim) shape.
        """
        # Transformation of the node
        x_before = x
        x = self.node_before_lin(x)
        x, xk = torch.chunk(x, 2, dim=-1)

        # Transformation of the coefficient vectors
        cji = self.coeffs_before_lin(cji)
        if self.add_valence:
            cji, ckj = torch.split(cji, [2 * self.conv_dim, self.conv_dim], dim=-1)
        else:
            cji, ckj = torch.chunk(cji, 2, dim=-1)

        # cutoff
        if cutoff_w is not None:
            robs = robs * cutoff_w.unsqueeze(-1)

        # triple conv
        ckj = ckj[edge_idx_kj]
        ckj = F.normalize(ckj, dim=-1)
        # LCAO weight: summation of all orbitals multiplied by coefficient vectors
        three_body_orbs = torch.einsum("ed,edh->eh", shbs * robs[edge_idx_kj], ckj)
        three_body_orbs = F.normalize(three_body_orbs, dim=-1)
        # multiply node embedding
        xk = torch.sigmoid(xk[tri_idx_k])
        three_body_w = three_body_orbs * xk
        three_body_w = self.three_lin(scatter(three_body_w, edge_idx_ji, dim=0, dim_size=robs.size(0)))
        # threebody orbital information is injected to the coefficient vectors
        cji = cji + cji * three_body_w.unsqueeze(1)
        cji = F.normalize(cji, dim=-1)
        if self.add_valence:
            cji, cji_valence = torch.chunk(cji, 2, dim=-1)

        # LCAO weight: summation of all orbitals multiplied by coefficient vectors
        lcao_w = torch.einsum("ed,edh->eh", robs, cji)

        if self.add_valence:
            # valence contribution
            if valence_mask is None:
                raise ValueError("valence_mask must be provided when add_valence=True")
            valence_w = torch.einsum("ed,edh->eh", robs, cji_valence * valence_mask)
            lcao_w = lcao_w + valence_w

        lcao_w = F.normalize(lcao_w, dim=-1)

        x = x_before + self.node_after_lin(
            scatter(lcao_w * self.node_lin(torch.cat([x[idx_i], x[idx_j]], dim=-1)), idx_i, dim=0)
        )

        return x


class LCAOOut(nn.Module):
    """The output layer of LCAONet.

    Three-layer neural networks to output desired physical property
    values.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        is_extensive: bool = True,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node embedding vectors.
            out_dim (int): the dimension of output property.
            is_extensive (bool): whether the output property is extensive or not. Defaults to `True`.
            activation (nn.Module): the activation function. Defaults to `nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None): the weight initialization function.
                Defaults to `None`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.is_extensive = is_extensive

        self.out_lin = nn.Sequential(
            activation,
            Dense(hidden_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim // 2, True, weight_init),
            activation,
            Dense(hidden_dim // 2, out_dim, False, weight_init),
        )

    def forward(self, x: Tensor, batch_idx: Tensor | None) -> Tensor:
        """Forward calculation of LCAOOut.

        Args:
            x (torch.Tensor): node embedding vectors with (n_node, hidden_dim) shape.
            batch_idx (torch.Tensor | None): the batch indices of nodes with (n_node) shape.

        Returns:
            torch.Tensor: the output property values with (n_batch, out_dim) shape.
        """
        out = self.out_lin(x)
        if batch_idx is not None:
            return scatter(out, batch_idx, dim=0, reduce="sum" if self.is_extensive else "mean")
        if self.is_extensive:
            return out.sum(dim=0, keepdim=True)
        else:
            return out.mean(dim=0, keepdim=True)


class LCAONet(BaseMPNN):
    """
    LCAONet - GNN including orbital interaction, physically motivatied by the LCAO method.
    """

    def __init__(
        self,
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
        postprocess: bool = False,
        **kwargs,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node embedding vectors. Defaults to `128`.
            coeffs_dim (int): the dimension of coefficient vectors. Defaults to `64`.
            conv_dim (int): the dimension of embedding vectors at convolution. Defaults to `64`.
            out_dim (int): the dimension of output property. Defaults to `1`.
            n_interaction (int): the number of interaction layers. Defaults to `3`.
            cutoff (float | None): the cutoff radius. Defaults to `None`.
                If specified, the basis functions are normalized within the cutoff radius.
                If `cutoff_net` is specified, the `cutoff` radius must be specified.
            cutoff_net (type[lcaonet.nn.cutoff.BaseCutoff] | None): the cutoff network. Defaults to `None`.
            bohr_radius (float): the bohr radius. Defaults to `0.529`.
            max_z (int): the maximum atomic number. Defaults to `36`.
            max_orb (str | None): the maximum orbital name like "2p". Defaults to `None`.
            elec_to_node (bool): whether to use electrons information to nodes embedding. Defaults to `True`.
            add_valence (bool): whether to add the effect of valence orbitals. Defaults to `False`.
            extend_orb (bool): whether to extend the basis set. Defaults to `False`.
                If `True`, convolution is performed including unoccupied orbitals of the ground state.
            is_extensive (bool): whether to predict extensive property. Defaults to `True`.
            activation (str): the name of activation function. Defaults to `"SiLU"`.
            weight_init (str | None): the name of weight initialization function. Defaults to `"glorotorthogonal"`.
            postprocess (bool): whether to use postprocess. Defaults to `False`.
        """
        super().__init__()
        wi: Callable[[Tensor], Tensor] | None = init_resolver(weight_init) if weight_init is not None else None
        act: nn.Module = activation_resolver(activation)

        self.hidden_dim = hidden_dim
        self.coeffs_dim = coeffs_dim
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.n_interaction = n_interaction
        self.cutoff = cutoff
        if cutoff_net is not None and cutoff is None:
            raise ValueError("cutoff must be specified when cutoff_net is used")
        self.cutoff_net = cutoff_net
        self.elec_to_node = elec_to_node
        self.add_valence = add_valence
        self.postprocess = postprocess

        # calc basis layers
        self.rob = RadialOrbitalBasis(cutoff, bohr_radius, max_z=max_z, max_orb=max_orb)
        self.shb = SphericalHarmonicsBasis(cutoff, bohr_radius, max_z=max_z, max_orb=max_orb)
        if cutoff_net:
            self.cn = cutoff_net(cutoff)  # type: ignore

        # node and coefficient embedding layers
        self.node_z_embed_dim = 64  # fix
        self.node_e_embed_dim = 64 if elec_to_node else 0  # fix
        self.e_embed_dim = self.coeffs_dim + self.node_e_embed_dim
        self.z_embed_dim = self.e_embed_dim + self.node_z_embed_dim
        self.z_embed = EmbedZ(embed_dim=self.z_embed_dim, max_z=max_z)
        self.e_embed = EmbedElec(self.e_embed_dim, max_z, max_orb, extend_orb)
        self.node_embed = EmbedNode(hidden_dim, self.node_z_embed_dim, elec_to_node, self.node_e_embed_dim, act, wi)
        if add_valence:
            self.valence_mask = ValenceMask(conv_dim, max_z)

        # interaction layers
        self.int_layers = nn.ModuleList(
            [LCAOConv(hidden_dim, coeffs_dim, conv_dim, add_valence, act, wi) for _ in range(n_interaction)]
        )

        # output layer
        self.out_layer = LCAOOut(hidden_dim, out_dim, is_extensive, act, wi)

        if postprocess:
            self.pp_layer = PostProcess(out_dim=out_dim, is_extensive=is_extensive, **kwargs)

    def forward(self, batch: Batch) -> Tensor:
        """Forward calculation of LCAONet.

        Args:
            batch (Batch): the input graph batch data.

        Returns:
            torch.Tensor: the output property with (n_batch, out_dim) shape.
        """
        batch_idx: Tensor | None = batch.get(DataKeys.Batch_idx)
        pos = batch[DataKeys.Position]
        z = batch[DataKeys.Atom_numbers]
        idx_i, idx_j = batch[DataKeys.Edge_idx]

        # get triplets
        (
            _,
            _,
            tri_idx_i,
            tri_idx_j,
            tri_idx_k,
            edge_idx_kj,
            edge_idx_ji,
        ) = self.get_triplets(batch)

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # calc angles each triplets
        pos_j = pos[tri_idx_j]
        vec_ji, vec_jk = pos[tri_idx_i] - pos_j, pos[tri_idx_k] - pos_j
        inner = (vec_ji * vec_jk).sum(dim=-1)
        outter = torch.cross(vec_ji, vec_jk).norm(dim=-1)
        # arctan is more stable than arccos
        angle = torch.atan2(outter, inner)

        # calc basis
        robs = self.rob(distances)
        shbs = self.shb(distances, angle, edge_idx_kj)
        cutoff_w = self.cn(distances) if self.cutoff_net else None

        # calc node and coefficient embedding vectors
        z_embed = self.z_embed(z)
        z_embed1, z_embed2 = torch.split(z_embed, [self.e_embed_dim, self.node_z_embed_dim], dim=-1)

        e_embed = self.e_embed(z, z_embed1)
        e_embed1, e_embed2 = torch.split(e_embed, [self.coeffs_dim, self.node_e_embed_dim], dim=-1)

        cji = e_embed1[idx_j] + e_embed1[idx_i] * e_embed1[idx_j]
        x = self.node_embed(z_embed2, e_embed2)

        # valence mask coefficients
        valence_mask: Tensor | None = self.valence_mask(z)[idx_j] if self.add_valence else None

        # calc interaction
        for inte in self.int_layers:
            x = inte(x, cji, valence_mask, cutoff_w, robs, shbs, idx_i, idx_j, tri_idx_k, edge_idx_kj, edge_idx_ji)

        # output
        out = self.out_layer(x, batch_idx)

        if self.postprocess:
            out = self.pp_layer(out, z, batch_idx)

        return out
