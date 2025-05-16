import torch
from torch.func import functional_call
from tqdm import tqdm

# -------- HVP over dataset -------------------------------------------

def hvp_dataset(model, criterion, params, vector, dataloader,
                main_device, hvp_fn):
    total = {k: torch.zeros_like(p) for k, p in params.items()}
    total_samples = 0
    for batch in tqdm(dataloader, desc="HVP", leave=False):
        batch = {k: v.to(main_device) for k, v in batch.items()}
        def batch_loss(p):
            outputs = functional_call(model, p,
                (batch["input_ids"], batch["attention_mask"]))
            return outputs.logits.sum()
        hvp_batch = hvp_fn(batch_loss, params, vector)
        bs = batch["input_ids"].size(0)
        with torch.no_grad():
            for k in total:
                total[k] += hvp_batch[k].detach() * bs
        total_samples += bs
        del hvp_batch
    return {k: total[k] / total_samples for k in total}, None

# -------- chunk helpers ----------------------------------------------

def chunked_dot(x_chunks, y_chunks):
    sums = [torch.sum(x.float() * y_chunks[n].float()).float()
            for n, x in x_chunks.items()]
    return torch.sum(torch.stack(sums))

def chunked_norm(x_chunks):
    return torch.sqrt(chunked_dot(x_chunks, x_chunks))

def chunked_saxpy_(alpha, x_chunks, y_chunks):
    for n, x in x_chunks.items():
        y_chunks[n].sub_((x.float() * alpha).to(y_chunks[n].dtype))
