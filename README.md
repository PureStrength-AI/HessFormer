# HessFormer

Distributed Hessian spectral density estimation for Transformer models,
built on top of PyTorch, HuggingFace `transformers`.

```python
import hessformer as hf

spec = hf.estimate_spectrum(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    dataset="JeanKaddour/minipile",
    num_iter=100,
)
spec.plot()
```
