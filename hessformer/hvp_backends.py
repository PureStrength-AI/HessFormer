from torch.func import grad, jvp
import torch

# ---------------------------------------------------------------------------#
#  Hessian–vector back-ends                                                  #
# ---------------------------------------------------------------------------#

def hvp_forward_over_reverse(loss_fn, params, v):
    """
    Forward-over-reverse:  ⟨H, v⟩ = JVP(grad(loss), v)
    """
    grad_fn = grad(loss_fn)                    # ∇ₚ ℒ           (reverse mode)
    _, hvp_pytree = jvp(grad_fn, (params,), (v,))  # JVP in dir v  (forward mode)
    return hvp_pytree                          #  → dict/ PyTree


def hvp_reverse_over_forward(loss_fn, params, v):
    """
    Reverse-over-forward:  compute directional derivative with JVP,
    then back-prop once more.
    """
    def loss_jvp(p):
        _, dir_loss = jvp(loss_fn, (p,), (v,))     # ⟨∇ℒ, v⟩
        return dir_loss

    return grad(loss_jvp)(params)              # ∇ₚ⟨∇ℒ, v⟩ = Hv


def hvp_reverse_over_reverse(loss_fn, params, v):
    """
    Reverse-over-reverse:  two reverse-mode passes.
    """
    grad_fn = grad(loss_fn)                    # ∇ℒ

    def grad_dot_v(p):
        g = grad_fn(p)
        return sum(torch.sum(g[k] * v[k]) for k in g)   # ⟨∇ℒ, v⟩

    return grad(grad_dot_v)(params)            # ∇ₚ⟨∇ℒ, v⟩ = Hv


# Registry so the high-level driver can look them up by name
BACKENDS = {
    "forward_reverse": hvp_forward_over_reverse,
    "reverse_forward": hvp_reverse_over_forward,
    "reverse_reverse": hvp_reverse_over_reverse,
}

def get_hvp_fn(name: str):
    try:
        return BACKENDS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend '{name}'. Choose from: {', '.join(BACKENDS)}"
        ) from e
