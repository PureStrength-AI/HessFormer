## README.md

## Cite

If you use HessFormer, please cite:

```bibtex
@misc{granziol2025hessformerhessiansfoundationscale,
  title={HessFormer: Hessians at Foundation Scale},
  author={Diego Granziol},
  year={2025},
  eprint={2505.11564},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.11564},
}
```

# lanczos_spectrum

A tiny, single-file package for **approximate Hessian spectral analysis** of very large Transformer language-models via the symmetric Lanczos algorithm.

* **Multi-GPU safe** – works with `device_map="auto"` and BF16 weights.  
* **Pluggable HVP back-ends** – choose between three automatic-differentiation layouts.  
* **One call** → Lanczos basis **Q**, tridiagonal **T**, eigenvalues, λ\_{min/max}, and γ₀² weights.

---

## Installation

```bash
git clone <repo-url>
cd lanczos_spectrum
pip install -e .
````

---

## Quick start

```python
import datasets, torch
from transformers import AutoModelForCausalLM
from lanczos_spectrum import run_lanczos, HvpMode

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train[:0.2%]")

out = run_lanczos(model, data,
                  num_iter=30,
                  hvp_mode=HvpMode.REV_FWD,   # default
                  batch_size=2)

print("λ_max:", out["spec"]["lambda_max"])
```

During execution you’ll see two progress bars:

* **HVP computation** – loops over the dataset and averages $H·v$.
* **Lanczos** – outer iteration up to `num_iter`.

---

## API

```python
run_lanczos(model,
            dataset,                 # HF dataset or list[str]
            num_iter=30,
            hvp_mode=HvpMode.REV_FWD,
            ortho=True,              # re-orthogonalise basis
            batch_size=2,
            tokenizer_name=None,     # auto-detect from model if None
            max_length=64)
```

**Returns**

```python
{
  "q":   list[list[Tensor]],   # Lanczos basis vectors (length = k_used)
  "T":   Tensor(k,k),          # real symmetric tridiagonal
  "spec": {
      "eigvals":    Tensor(k),
      "gammas":     Tensor(k),  # (first-basis coeffs)²
      "lambda_min": float,
      "lambda_max": float
  }
}
```

---

## `hvp_mode`

Three back-ends are available (`REV_FWD`, `FWD_REV`, `REV_REV`).
See the included **cheat-sheet** in `docs/hvp_modes.md` if you need guidance on which to pick.

---

## Tips

* **Memory** – 70 B models need multiple A100s; reduce `num_iter` or dataset size otherwise.
* **Density plots** – a simple stem plot of `eigvals` vs `gammas` gives a spiky spectral picture; summing `gammas` into bins yields a smooth density estimate.
* **Stability** – keep `ortho=True` unless you need absolute peak throughput.

Happy curvature-surfing 🚀
