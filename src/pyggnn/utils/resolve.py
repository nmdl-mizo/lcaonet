from __future__ import annotations

from collections.abc import Callable
from typing import Any
from inspect import getmembers, isfunction

import torch
from torch.nn.init import calculate_gain
from torch_geometric.nn.inits import glorot, glorot_orthogonal

from pyggnn.nn.activation import ShiftedSoftplus, Swish

__all__ = ["activation_resolver", "activation_gain_resolver", "init_resolver"]


def _normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def _resolver(
    query: Any | str,
    classes: list[Any],
    base_cls: Any | None = None,
    return_initialize: bool = True,
    **kwargs,
) -> Callable | Any:
    # query is a string
    if isinstance(query, str):
        for cls in classes:
            if _normalize_string(cls.__name__.lower()) == query:
                # deprecated function is not used
                if "deprecated" in cls.__str__():
                    continue
                if not return_initialize:
                    return cls
                obj = cls(**kwargs)
                assert callable(obj)
                return obj
    # query is some type
    elif isinstance(query, type):
        if query in classes:
            if not return_initialize:
                return query
            obj = query(**kwargs)
            assert callable(obj)
            return obj
        if issubclass(query, base_cls) and base_cls is not None:
            if not return_initialize:
                return query
            obj = query(**kwargs)
            assert callable(obj)
            return obj
        elif isinstance(query, base_cls) and base_cls is not None:
            if not return_initialize:
                return query
            obj = query(**kwargs)
            assert callable(obj)
            return obj
    # query is callable
    elif isinstance(query, Callable):
        if query in classes:
            assert callable(query)
            return query
    else:
        raise ValueError(f"{query} must be str or type or class")
    raise ValueError(f"{query} not found")


def activation_resolver(query: torch.nn.Module | str = "relu", **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(query, str):
        query = _normalize_string(query)
    base_cls = torch.nn.Module
    # activation classes
    acts = [
        act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    # add Swish and ShiftedSoftplus
    acts += [Swish, ShiftedSoftplus]
    return _resolver(query, acts, base_cls, **kwargs)


def activation_gain_resolver(query: torch.nn.Module | str = "relu", **kwargs) -> float:
    if isinstance(query, str):
        query = _normalize_string(query)
    base_cls = torch.nn.Module
    # activation classes
    acts = [
        act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    # add Swish and ShiftedSoftplus
    acts += [Swish, ShiftedSoftplus]
    gain_dict = {
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "relu": "relu",
        "selu": "selu",
        "leakyrelu": "leaky_relu",
        # swish using sigmoid gain
        "swish": "sigmoid",
        # shifted softplus using linear gain
        "shiftedsoftplus": "linear",
    }
    # if query is not found, return "linear" gain
    try:
        nonlinearity = gain_dict.get(
            _resolver(query, acts, base_cls, False).__name__.lower(), "linear"
        )
    except ValueError:
        nonlinearity = "linear"
    return calculate_gain(nonlinearity, **kwargs)


def init_resolver(query: Callable | str = "orthogonal") -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(query, str):
        query = _normalize_string(query)

    funcs = [f[1] for f in getmembers(torch.nn.init, isfunction)]
    # add torch_geometric.nn.inits
    funcs += [glorot, glorot_orthogonal]
    # Since the list contains callable instead of class, return without initialize.
    return _resolver(query, funcs, return_initialize=False)


def init_param_resolver(query: Callable[[torch.Tensor], torch.Tensor]) -> tuple[str]:
    params = query.__code__.co_varnames[: query.__code__.co_argcount]
    return params
