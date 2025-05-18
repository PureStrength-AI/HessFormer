"""
lanczos_spectrum
~~~~~~~~~~~~~~~~
One-file package for approximate spectral analysis of very large
Transformer models with the Lanczos algorithm.

Input  :  (model, dataset, n_iter, ortho?, hvp_mode)
Output :  {"q": [Tensor …],           # Lanczos basis (len = k_used)
           "T": Tensor(k,k),          # tridiagonal β/α matrix
           "spec": {"eigvals": Tensor,
                     "gammas" : Tensor,
                     "lambda_min": float,
                     "lambda_max": float}}
"""

from __future__ import annotations
import enum, math, types, warnings, itertools
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.func import grad, jvp, functional_call
from tqdm import tqdm
from transformers import AutoTokenizer

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# --------------------------------------------------------------------- #
#  HVP back-end choices
# --------------------------------------------------------------------- #
class HvpMode(str, enum.Enum):
    FWD_REV = "forward_reverse"     # forward-over-reverse
    REV_FWD = "reverse_forward"     # reverse-over-forward   (fast ∣ default)
    REV_REV = "reverse_reverse"     # double-reverse

def _hvp_fwd_rev(loss_fn, p, v):        # forward over reverse
    return jvp(grad(loss_fn), (p,), (v,))[1]

def _hvp_rev_fwd(loss_fn, p, v):        # reverse over forward
    def loss_jvp(pp):
        return jvp(loss_fn, (pp,), (v,))[1]
    return grad(loss_jvp)(p)

def _hvp_rev_rev(loss_fn, p, v):        # reverse over reverse
    def grad_dot_v(pp):
        g = grad(loss_fn)(pp)
        return sum([(g[k] * v[k]).sum() for k in g])
    return grad(grad_dot_v)(p)

_HVP_DISPATCH = {
    HvpMode.FWD_REV   : _hvp_fwd_rev,
    HvpMode.REV_FWD   : _hvp_rev_fwd,
    HvpMode.REV_REV   : _hvp_rev_rev,
}

# --------------------------------------------------------------------- #
#  Helper: flatten / unflatten param pytree       (dict[str,Tensor] <-> List)
# --------------------------------------------------------------------- #
def _flatten(params: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[str]]:
    keys, flat = zip(*params.items())
    return list(flat), list(keys)

def _unflatten(flat: List[torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in zip(keys, flat)}

# --------------------------------------------------------------------- #
#  Hessian-vector product operator averaged over a dataloader
# --------------------------------------------------------------------- #
class _HvpOperator:
    def __init__(
        self,
        model: torch.nn.Module,
        hvp_fn,
        params_k: List[str],
        dataloader: DataLoader,
        main_device: torch.device,
    ):
        self.model, self.dataloader = model, dataloader
        self.hvp_fn, self.params_k = hvp_fn, params_k
        self.main_device = main_device

    def __call__(self, v_dict):
        total = {k: torch.zeros_like(t) for k, t in v_dict.items()}
        n = 0
    
        for batch in tqdm(self.dataloader, desc="HVP computation"):
            batch = {bk: bv.to(self.main_device) for bk, bv in batch.items()}
    
            def batch_loss(p):
                out = functional_call(self.model, p,
                                      (batch["input_ids"], batch.get("attention_mask")))
                return out.logits.float().mean()
    
            hvp_batch = self.hvp_fn(batch_loss, v_dict, v_dict)   # dict of chunks
    
            bs = batch["input_ids"].size(0)
            with torch.no_grad():
                for k in total:
                    total[k] += hvp_batch[k].detach().to(total[k].device) * bs
            n += bs
    
        for k in total:
            total[k] /= float(n)
        return total

# --------------------------------------------------------------------- #
#  Main public utility
# --------------------------------------------------------------------- #
@torch.no_grad()
def run_lanczos(
    model: torch.nn.Module,
    dataset,                           # HF dataset OR list[str]
    num_iter: int                     = 30,
    hvp_mode: HvpMode                 = HvpMode.REV_FWD,
    ortho: bool                       = True,
    batch_size: int                   = 2,
    tokenizer_name: str | None        = None,
    max_length: int                   = 64,
):
    """
    Drive the classic symmetric Lanczos recurrence to approximate
    the top-k part of the Hessian spectrum.

    Returns a dictionary with Lanczos basis, tridiagonal matrix and
    spectral info (eigenvalues + first-basis gammas).

    NOTE
    ----
    * Make sure the model is in `.eval()` mode.
    * This is memory-heavy; you probably want `torch_dtype=bfloat16`.
    """
    if tokenizer_name is None:
        tokenizer_name = getattr(model, "name_or_path", None)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 1. build dataloader ------------------------------------------------
    if isinstance(dataset, list) and isinstance(dataset[0], str):
        # treat as raw strings
        ds = [{"text": t} for t in dataset]
    else:
        ds = dataset

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=max_length)
    ds_tok = ds.map(_tok, batched=True) if hasattr(ds, "map") else ds
    ds_tok.set_format(type="torch", columns=["input_ids"])
    dl = DataLoader(ds_tok, batch_size=batch_size, shuffle=False)

    # 2. prep params -----------------------------------------------------
    model_device = next(model.parameters()).device
    params_dict = dict(model.named_parameters())
    params_flat, params_keys = _flatten(params_dict)

    # pick hvp backend ---------------------------------------------------
    hvp_fn = _HVP_DISPATCH[HvpMode(hvp_mode)]
    hvp_op = _HvpOperator(model, hvp_fn, params_keys, dl, model_device)

    # 3. initialise v₀ (random unit vector) -----------------------------
    q_prev, q_curr = [torch.zeros_like(p) for p in params_flat], [
        torch.randn_like(p) for p in params_flat
    ]
    _normalise(q_curr)

    # outputs ------------------------------------------------------------
    q_basis: List[List[torch.Tensor]] = []
    alphas, betas = [], []

    for k in range(num_iter):
        # store current basis vector
        q_basis.append([t.clone() for t in q_curr])

        #   r  = H q_k
        r_dict = hvp_op(_unflatten(q_curr, params_keys))
        r_flat, _ = _flatten(r_dict)

        #   alpha_k = q_kᵀ r
        alpha_k = _dot(q_curr, r_flat).item()
        alphas.append(alpha_k)

        #   r  ← r  − alpha_k q_k − beta_{k-1} q_{k-1}
        _saxpy(-alpha_k, q_curr, r_flat)
        if k > 0:
            _saxpy(-betas[-1], q_prev, r_flat)

        beta_k = _norm(r_flat).item()
        betas.append(beta_k)

        if beta_k < 1e-10:
            # converged
            break

        # orthonormalise if requested (simple re-orth step)
        if ortho:
            _orthonormalise(r_flat, q_basis)

        # rotate vectors
        q_prev = [t.clone() for t in q_curr]
        q_curr = [t.clone() for t in r_flat]
        _normalise(q_curr)

    # build tridiagonal T -----------------------------------------------
    k_used = len(alphas)
    T = torch.zeros(k_used, k_used, device=model_device)
    for i,(a,b) in enumerate(zip(alphas, betas)):
        T[i,i] = a
        if i+1 < k_used:
            T[i+1,i] = T[i,i+1] = b

    eigvals, eigvecs = torch.linalg.eigh(T)
    gammas   = eigvecs[0]**2

    spec = dict(eigvals=eigvals, gammas=gammas,
                lambda_min=float(eigvals.min()),
                lambda_max=float(eigvals.max()))

    return {"q": q_basis, "T": T, "spec": spec}

# --------------------------------------------------------------------- #
#  ––– internal linear-algebra helpers –––––––––––––––––––––––––––––––– #
# --------------------------------------------------------------------- #
# -------------------------------------------------------------
#   helpers   (place at the bottom of lanczos_spectrum/__init__.py)
# -------------------------------------------------------------
# ---------------------------------------------------------------------
# safe helpers for any device-sharded model
# ---------------------------------------------------------------------
def _dot(a, b):
    main = a[0].device
    acc  = torch.zeros((), dtype=torch.float32, device=main)
    for x, y in zip(a, b):
        acc += (x.float() * y.float()).sum().to(main)
    return acc                                   # scalar tensor on `main`

def _norm(v):
    return _dot(v, v).sqrt()                     # scalar on `main`

def _normalise(v):
    vnorm = _norm(v)                            # on `main`
    # OPTION 1: move to each device
    for t in v:
        t.div_(vnorm.to(t.device))

def _saxpy(alpha, x, y):
    for xi, yi in zip(x, y):
        yi.add_(xi, alpha=float(alpha))          # Python float broadcasts

def _orthonormalise(v, basis):
    for q in basis:                     # Gram–Schmidt
        _saxpy(-_dot(q, v), q, v)
    _normalise(v)
