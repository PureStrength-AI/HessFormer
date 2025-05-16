import functorch
from typing import Dict, Callable

def hvp_reverse_forward(f, params, v):
    return functorch.hvp(f, params, v)

def hvp_reverse_reverse(f, params, v):
    return functorch.hvp(f, params, v, forward=False, reverse=True)

def hvp_forward_reverse(f, params, v):
    return functorch.hvp(f, params, v, forward=True, reverse=False)

BACKENDS = {
    "reverse_forward": hvp_reverse_forward,
    "reverse_reverse": hvp_reverse_reverse,
    "forward_reverse": hvp_forward_reverse,
}

def get_hvp_fn(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Choose from {list(BACKENDS)}")
    return BACKENDS[name]
