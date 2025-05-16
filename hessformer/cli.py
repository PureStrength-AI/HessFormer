import argparse, json, pathlib
from .core import estimate_spectrum

def main():
    p = argparse.ArgumentParser("HessFormer CLI")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="JeanKaddour/minipile")
    p.add_argument("-k", "--iters", type=int, default=50)
    p.add_argument("--backend", default="reverse_forward",
                   choices=["reverse_forward", "reverse_reverse", "forward_reverse"])
    p.add_argument("--no-ortho", dest="ortho", action="store_false")
    p.add_argument("--out", default="spectrum.json")
    args = p.parse_args()
    res = estimate_spectrum(
        model=args.model,
        dataset=args.dataset,
        num_iter=args.iters,
        hvp_backend=args.backend,
        orthonormalize=args.ortho
    )
    data = dict(evals=res.evals.tolist(), weights=res.weights.tolist(), meta=res.meta)
    pathlib.Path(args.out).write_text(json.dumps(data))
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
