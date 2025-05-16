from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Union

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    PreTrainedModel, PreTrainedTokenizerBase,
)
from torch.utils.data import Dataset
from .data import build_dataloader
from .hvp_backends import get_hvp_fn
from .lanczos import stochastic_lanczos_quadrature


@dataclass
class SpectralResult:
    vectors: List[Dict[str, torch.Tensor]]
    tridiag: torch.Tensor
    evals: torch.Tensor
    weights: torch.Tensor
    meta: Dict[str, Union[str, int, float]]

    def plot(self, log: bool = True):
        import matplotlib.pyplot as plt
        plt.bar(self.evals.cpu(), self.weights.cpu(),
                width=(self.evals[1] - self.evals[0]).abs().item()
                if len(self.evals) > 1 else 1.0,
                align="center")
        if log:
            plt.yscale("log")
        plt.xlabel("Eigenvalue λ")
        plt.ylabel("ρ(λ)")
        plt.title("Hessian spectrum")
        plt.show()


# ---------------------------------------------------------------------

def _load_model_and_tokenizer(
    model: Union[str, PreTrainedModel],
    dtype=torch.bfloat16,
    device_map: str | None = "auto",
):
    if isinstance(model, str):
        model_ = AutoModelForCausalLM.from_pretrained(
            model, device_map=device_map, torch_dtype=dtype)
    else:
        model_ = model
    model_.config.use_cache = False
    model_.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(model_, "name_or_path", model))
    return model_.eval(), tokenizer


def estimate_spectrum(
    model: Union[str, PreTrainedModel],
    dataset: Union[str, Dataset, Sequence[str]],
    num_iter: int = 50,
    hvp_backend: Literal[
        "reverse_reverse", "reverse_forward", "forward_reverse"
    ] = "reverse_forward",
    orthonormalize: bool = True,
    batch_size: int = 2,
    max_length: int = 64,
    dtype=torch.bfloat16,
    device_map: str | None = "auto",
    seed: int = 42,
) -> SpectralResult:
    """High‑level API: return Hessian spectral estimate via SLQ."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_, tokenizer = _load_model_and_tokenizer(model, dtype, device_map)
    dataloader = build_dataloader(
        dataset, tokenizer, batch_size=batch_size, max_length=max_length)

    hvp_fn = get_hvp_fn(hvp_backend)

    vectors, T = stochastic_lanczos_quadrature(
        model=model_,
        dataloader=dataloader,
        hvp_fn=hvp_fn,
        k=num_iter,
        orthonormalize=orthonormalize,
    )

    evals, vecs = torch.linalg.eigh(T)
    e1 = torch.zeros_like(evals); e1[0] = 1.0
    weights = (vecs.T @ e1).pow(2)

    return SpectralResult(
        vectors=vectors,
        tridiag=T,
        evals=evals,
        weights=weights,
        meta=dict(
            backend=hvp_backend,
            k=num_iter,
            ortho=orthonormalize,
            device=next(model_.parameters()).device.type,
        ),
    )
