import torch, random, math
from typing import Dict, List
from .utils import hvp_dataset, chunked_dot, chunked_norm, chunked_saxpy_

class _HvpOperator:
    def __init__(self, model, params, device, hvp_fn, dataloader):
        self.model, self.params, self.device = model, params, device
        self.hvp_fn, self.dataloader = hvp_fn, dataloader

    def __call__(self, vector):
        hvp, _ = hvp_dataset(
            self.model, None, self.params, vector,
            self.dataloader, self.device, self.hvp_fn)
        return hvp

# --------------------------------------------------------------------

def _random_unit_vector(params, device):
    numels = {n: p.numel() for n, p in params.items()}
    total = sum(numels.values())
    idx = random.randrange(total)
    vec = {}
    start = 0
    for n, p in params.items():
        num = numels[n]
        v = torch.zeros(num, dtype=p.dtype, device=device)
        if start <= idx < start + num:
            v[idx - start] = 1.0
        vec[n] = v
        start += num
    # normalisation
    norm = torch.sqrt(sum([(v.float() ** 2).sum() for v in vec.values()]))
    for n in vec:
        vec[n].div_(norm)
    return vec

def _reorthogonalise(basis: List[Dict[str, torch.Tensor]], q):
    for qi in basis:
        coeff = chunked_dot(qi, q)
        chunked_saxpy_(coeff, qi, q)
    norm = chunked_norm(q)
    for n in q:
        q[n].div_(norm)

# --------------------------------------------------------------------

def stochastic_lanczos_quadrature(
    model, dataloader, hvp_fn, k: int = 50, orthonormalize: bool = True):
    params = dict(model.named_parameters())
    device = next(model.parameters()).device
    op = _HvpOperator(model, params, device, hvp_fn, dataloader)

    q_curr = _random_unit_vector(params, device)
    basis, alphas, betas = [], [], []

    for it in range(k):
        basis.append(q_curr)
        shaped_q = {n: q.view(p.shape) for (n, p), q in zip(params.items(), q_curr.values())}
        z = op(shaped_q)
        alpha = chunked_dot(q_curr, z).item()
        alphas.append(alpha)
        if it:
            chunked_saxpy_(beta, q_prev, z)
        chunked_saxpy_(alpha, q_curr, z)
        beta = chunked_norm(z).item()
        betas.append(beta)
        if beta < 1e-12:
            break
        q_prev = q_curr
        q_curr = {n: r / beta for n, r in z.items()}
        if orthonormalize:
            _reorthogonalise(basis, q_curr)

    # assemble tridiagonal
    m = len(alphas)
    T = torch.zeros(m, m, device=device)
    for i, a in enumerate(alphas):
        T[i, i] = a
        if i + 1 < m:
            T[i, i+1] = T[i+1, i] = betas[i+1] if i+1 < len(betas) else betas[-1]
    return basis, T
