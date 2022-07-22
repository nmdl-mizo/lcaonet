from typing import Optional, Union, Any, List

from pyggnn.nn.activation import ShiftedSoftplus, Swish

__all__ = ["activation_resolver", "activation_gain_resolver"]


def _normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def _resolver(
    query: Union[Any, str],
    classes: List[Any],
    base_cls: Optional[Any] = None,
    return_cls: bool = False,
    **kwargs,
):
    if isinstance(query, str):
        query = _normalize_string(query)
    if isinstance(query, str):
        for cls in classes:
            if cls.__name__.lower() == query:
                if return_cls:
                    return cls
                obj = cls(**kwargs)
                assert callable(obj)
                return obj
    elif isinstance(query, type):
        if query in classes:
            if return_cls:
                return cls
            obj = query(**kwargs)
            assert callable(obj)
            return obj
    elif isinstance(query, base_cls):
        if return_cls:
            return cls
        obj = query(**kwargs)
        assert callable(obj)
        return obj
    else:
        raise ValueError("query must be str or type or class")
    raise ValueError("query not found")


def activation_resolver(query: Union[Any, str] = "relu", **kwargs):
    import torch

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


def activation_gain_resolver(query: Union[Any, str] = "relu") -> str:
    import torch

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
    # if not found, return "linear"
    return gain_dict.get(
        _resolver(query, acts, base_cls, True).__name__.lower(), "linear"
    )
