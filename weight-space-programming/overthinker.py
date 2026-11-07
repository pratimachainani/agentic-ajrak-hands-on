#!/usr/bin/env python3
"""Fix an overthinking model by editing its weights, no training.

A small, hands-on version of the Qwen fix in "Sparks of Weight-Space Programming"
(Omar Khattab's group, MIT CSAIL, 2 Nov 2026). Four steps:

  observe    run the model on practice problems; count answers that run out of tokens
  direction  average the residual stream of looped vs finished answers; the difference is u
  patch      remove u from the top-k columns of one layer's MLP output matrix; save the model
  observe    run the patched model on the same problems; count again

Get the model first (ModelScope, no Hugging Face account needed):
  modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ./qwen3.5-0.8b

Usage (--model is the local folder, or any Hugging Face id):
  python overthinker.py observe   --model ./qwen3.5-0.8b
  python overthinker.py direction --model ./qwen3.5-0.8b --layer 15
  python overthinker.py patch     --model ./qwen3.5-0.8b --layer 15 --top 64
  python overthinker.py observe   --model ./qwen-patched
"""
import argparse, json, pathlib, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).parent
RUNS = HERE / "runs"; RUNS.mkdir(exist_ok=True)


def device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except (ValueError, KeyError):
        # Qwen3.5 is listed as image-text-to-text; load the full model and use only its text path
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float32)
    return tok, model.to(device()).eval()


def layers_of(model):
    """The list of decoder layers (transformer blocks), whatever the architecture calls it."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."): obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise SystemExit("Could not find the decoder layers. Run: python -c \"from transformers import AutoModelForCausalLM as M; print(M.from_pretrained('<model>'))\"")


def mlp_out(layer):
    """The MLP's output matrix (Linear: intermediate -> hidden). Its columns are what the MLP can add to the residual stream."""
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    for name in ("down_proj", "c_proj", "fc2", "w2"):
        if mlp is not None and hasattr(mlp, name): return getattr(mlp, name)
    raise SystemExit(f"Could not find the MLP output matrix in this layer:\n{layer}")


def prompts(tok, n, file="problems.json"):
    qs = json.load(open(HERE / file))[:n]
    out = []
    for q in qs:
        msgs = [{"role": "user", "content": q + "\nPut the final answer in \\boxed{}."}]
        try:
            out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        except TypeError:
            out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def observe(a):
    tok, model = load(a.model)
    ps = prompts(tok, a.n)
    rows, t0 = [], time.time()
    for i in range(0, len(ps), a.batch):
        enc = tok(ps[i:i + a.batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                 pad_token_id=tok.pad_token_id)
        for j, seq in enumerate(gen):
            new = seq[enc["input_ids"].shape[1]:]
            ended = bool((new == tok.eos_token_id).any() or (new == tok.pad_token_id).any())
            rows.append({"prompt": ps[i + j], "output": tok.decode(new, skip_special_tokens=True), "ran_out": not ended})
        print(f"  {len(rows)}/{len(ps)} done · {time.time() - t0:.0f}s")
    tag = pathlib.Path(a.model).name
    json.dump(rows, open(RUNS / f"observe-{tag}.json", "w"), indent=1)
    report(rows, tag, a.max_tokens)


def report(rows, tag, limit):
    for r in rows:
        w = r["output"].split(); g = [tuple(w[i:i + 6]) for i in range(len(w) - 6)]
        r["repeat"] = round(1 - len(set(g)) / max(1, len(g)), 3)      # share of 6-word phrases it has already said
        r["answered"] = bool(re.search(r"\\boxed\{[^{}]*\d[^{}]*\}", r["output"]))  # a boxed answer with a number in it
    print(f"\n{tag}: {sum(r['ran_out'] for r in rows)} / {len(rows)} answers ran out of tokens (limit {limit})")
    print(f"  gave a boxed answer: {sum(r['answered'] for r in rows)} / {len(rows)}")
    print(f"  repeating itself (over 20% repeated phrases): {sum(r['repeat'] > 0.2 for r in rows)} / {len(rows)}")


def score(a):
    tag = pathlib.Path(a.model).name
    report(json.load(open(RUNS / f"observe-{tag}.json")), tag, "as run")
    print(f"saved runs/observe-{tag}.json")


def residual_after(model, enc, layer):
    """Run the model once and return the residual stream right after `layer` (tokens x hidden).
    Uses a hook and skips the vocabulary logits, so an 8,000-token answer fits in GPU memory."""
    got = {}
    def grab(mod, inp, out): got["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    hook = layers_of(model)[layer].register_forward_hook(grab)
    try:
        with torch.no_grad():
            try: model(**enc, logits_to_keep=1)
            except TypeError: model(**enc)
    finally:
        hook.remove()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return got["h"][0]


def loop_start(ids, n=8, times=3):
    """Index of the first token where some n-token phrase appears for the `times`-th time (the loop has begun)."""
    seen = {}
    for i in range(len(ids) - n):
        g = tuple(ids[i:i + n]); seen[g] = seen.get(g, 0) + 1
        if seen[g] == times: return i
    return len(ids)


def direction(a):
    """u = average residual stream while looping, minus a 'not looping' average.
    contrast 'finished': looped answers vs answers that finished (needs both kinds).
    contrast 'early-late': within each looped answer, its last tokens (going round in circles)
                           vs its first tokens of thinking (still reasoning). Works when every answer looped."""
    tok, model = load(a.model)
    rows = json.load(open(RUNS / f"observe-{pathlib.Path(a.model).name}.json"))
    n_loop = sum(r["ran_out"] for r in rows); n_fin = len(rows) - n_loop
    mode = a.contrast
    if mode == "auto": mode = "finished" if n_loop and n_fin else "early-late"
    if mode == "finished" and not (n_loop and n_fin):
        raise SystemExit("Need at least one looped and one finished answer for --contrast finished. Use --contrast early-late.")
    if mode == "early-late" and not n_loop:
        raise SystemExit("No answer ran out of tokens, so there is no loop to study. Lower --max-tokens and observe again.")
    pos, neg = [], []
    for r in rows:
        if mode == "early-late" and not r["ran_out"]: continue
        p_len = tok(r["prompt"], return_tensors="pt")["input_ids"].shape[1]
        enc = tok(r["prompt"] + r["output"], return_tensors="pt").to(model.device)
        h = residual_after(model, enc, a.layer)    # residual stream after layer `layer`, one vector per token
        if mode == "finished":
            (pos if r["ran_out"] else neg).append(h[-a.last:].mean(0).float().cpu())
        else:
            ans = h[p_len:]
            start = loop_start(enc["input_ids"][0, p_len:].tolist())   # first token where the answer starts repeating itself
            k = max(8, min(a.last, start))
            neg.append(ans[:k].mean(0).float().cpu())   # before the loop: still reasoning
            pos.append(ans[-a.last:].mean(0).float().cpu())  # the end: going round in circles
    u = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
    u = u / u.norm()
    torch.save({"u": u, "layer": a.layer, "contrast": mode, "pos": len(pos), "neg": len(neg)}, RUNS / "direction.pt")
    if mode == "early-late": print("  (start = tokens before each answer begins repeating itself)")
    what = (f"{len(pos)} looped vs {len(neg)} finished answers" if mode == "finished"
            else f"the last tokens vs the first tokens of {len(pos)} looped answers")
    print(f"stuck direction: {u.numel()} dims (the residual stream width), from {what}, after layer {a.layer}")
    print("saved runs/direction.pt")


def writers(layer, hidden, which):
    """Matrices in one layer that write into the residual stream.
    which='mlp': only the MLP output matrix. which='all': every Linear whose output is the residual width
    (the MLP output, and the attention or linear-attention output projection)."""
    if which == "mlp":
        return [("mlp", mlp_out(layer))]
    out_names = ("o_proj", "out_proj", "down_proj", "c_proj", "fc2", "w2")   # output projections only, never input ones
    return [(n, m) for n, m in layer.named_modules()
            if isinstance(m, torch.nn.Linear) and m.out_features == hidden and n.split(".")[-1] in out_names]


def patch(a):
    tok, model = load(a.model)
    d = torch.load(RUNS / "direction.pt")
    u = d["u"].to(model.device)
    hidden = u.numel()
    total = 0
    if a.top <= 0: a.top = 10**9
    layers = layers_of(model)
    picks = range(len(layers)) if a.layer == ["all"] else [int(x) for x in a.layer]
    for L in picks:
        for name, lin in writers(layers[L], hidden, a.writers):
            W = lin.weight.data                                  # shape [hidden, in]; each column writes into the residual stream
            scores = (u @ W).abs()                               # how strongly each column writes along u
            idx = scores.topk(min(a.top, W.shape[1])).indices    # --top 0 means every column
            cols = W[:, idx]
            W[:, idx] = cols - torch.outer(u, u @ cols)          # remove the u part of those columns, keep the rest
            total += cols.numel()
            print(f"layer {L} {name}: edited {len(idx)} of {W.shape[1]} columns ({cols.numel():,} weights)")
    n = sum(p.numel() for p in model.parameters())
    out = HERE / a.out
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"edited {total:,} of {n:,} weights ({100 * total / n:.4f}%) · saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("observe"); o.add_argument("--model", required=True); o.add_argument("--n", type=int, default=32)
    o.add_argument("--max-tokens", type=int, default=1024); o.add_argument("--batch", type=int, default=8)
    d = sp.add_parser("direction"); d.add_argument("--model", required=True); d.add_argument("--layer", type=int, default=15)
    d.add_argument("--last", type=int, default=256)
    d.add_argument("--contrast", choices=["auto", "finished", "early-late"], default="auto")
    p = sp.add_parser("patch"); p.add_argument("--model", required=True); p.add_argument("--layer", nargs="+", default=["15"], help="layer numbers, or: all")
    p.add_argument("--writers", choices=["mlp", "all"], default="mlp", help="mlp: MLP output only; all: every matrix that writes to the residual stream")
    p.add_argument("--top", type=int, default=64); p.add_argument("--out", default="qwen-patched")
    s = sp.add_parser("score"); s.add_argument("--model", required=True)
    a = ap.parse_args()
    {"observe": observe, "direction": direction, "patch": patch, "score": score}[a.cmd](a)
