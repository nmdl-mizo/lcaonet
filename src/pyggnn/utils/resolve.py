from typing import Callable, Optional, Union, Any, List
from inspect import getmembers, isfunction

import torch
from torch.nn.init import calculate_gain

from pyggnn.nn.activation import ShiftedSoftplus, Swish

__all__ = ["activation_resolver", "activation_gain_resolver", "init_resolver"]


def _normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def _resolver(
    query: Union[Any, str],
    classes: List[Any],
    base_cls: Optional[Any] = None,
    return_initialize: bool = True,
    **kwargs,
) -> Union[Callable, Any]:
    if isinstance(query, str):
        for cls in classes:
            if cls.__name__.lower() == query:
                if not return_initialize:
                    return cls
                obj = cls(**kwargs)
                assert callable(obj)
                return obj
    elif isinstance(query, type):
        if query in classes:
            if not return_initialize:
                return query
            obj = query(**kwargs)
            assert callable(obj)
            return obj
    elif issubclass(query, base_cls) and base_cls is not None:
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
    else:
        raise ValueError(f"{query} must be str or type or class")
    raise ValueError(f"{query} not found")


def activation_resolver(
    query: Union[torch.nn.Module, str] = "relu", **kwargs
) -> Callable:
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


def activation_gain_resolver(
    query: Union[torch.nn.Module, str] = "relu", **kwargs
) -> float:
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
    return calculate_gain(
        gain_dict.get(
            _resolver(query, acts, base_cls, False).__name__.lower(), "linear"
        ),
        **kwargs,
    )


def init_resolver(query: Union[torch.nn.Module, str] = "orthogonal"):
    if query[-1] != "_":
        query = _normalize_string(query)
        query += "_"

    funcs = [f[1] for f in getmembers(torch.nn.init, isfunction)]

    return _resolver(query, funcs, return_initialize=False)
