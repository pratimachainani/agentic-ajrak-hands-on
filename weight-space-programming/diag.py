#!/usr/bin/env python3
"""Diagnose the overthinker: where does the decision to STOP thinking live? (L02 hands-on, part 2)

  stopsignal  how likely the model is to close its thinking (</think>) at each point of saved answers
  easy        run easy questions (they finish), to get examples of the model deciding to stop
  finishdir   'finish' direction at every layer: residual just before </think> in finished answers
              minus residual late in looping answers
  steer       test a layer's finish direction at runtime (a hook, no weight change) on a few hard questions
  bake        write the best layer's finish direction into the weights (boost it in that layer's writers)

Small runs: --n 8 and --max-tokens 4096 by default, so each test takes minutes.
"""
import argparse, json, pathlib, re, time
import torch
from overthinker import load, layers_of, prompts, writers, RUNS, HERE


BOX = r"\\boxed\{[^{}]*\d[^{}]*\}"


def end_id(tok):
    i = tok.convert_tokens_to_ids("</think>")
    if i is None or i == tok.unk_token_id:
        raise SystemExit("This tokenizer has no </think> token.")
    return i


def gen(tok, model, ps, max_tokens, batch, stop_id):
    rows = []
    for i in range(0, len(ps), batch):
        enc = tok(ps[i:i + batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                 pad_token_id=tok.pad_token_id)
        for j, seq in enumerate(out):
            new = seq[enc["input_ids"].shape[1]:]
            closed = bool((new == stop_id).any())
            ended = bool((new == tok.eos_token_id).any() or (new == tok.pad_token_id).any())
            text = tok.decode(new, skip_special_tokens=False).replace(tok.pad_token or "\x00", "")
            rows.append({"prompt": ps[i + j], "output": text, "closed_thinking": closed, "ran_out": not ended})
    return rows


def summary(rows, label):
    def rep(o):
        w = o.split(); g = [tuple(w[k:k + 6]) for k in range(len(w) - 6)]
        return 1 - len(set(g)) / max(1, len(g))
    n = len(rows)
    boxed = sum(bool(re.search(BOX, r["output"])) for r in rows)
    reps = sum(rep(r["output"]) > 0.2 for r in rows)
    print(f"{label}: closed thinking {sum(r['closed_thinking'] for r in rows)}/{n} · ran out {sum(r['ran_out'] for r in rows)}/{n} · "
          f"boxed {boxed}/{n} · repeating {reps}/{n}")


def all_layers(model, enc):
    """Residual stream after every layer, in one forward pass (no vocabulary logits)."""
    got, hooks = {}, []
    for L, layer in enumerate(layers_of(model)):
        hooks.append(layer.register_forward_hook(lambda m, i, o, L=L: got.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()[0])))
    try:
        with torch.no_grad():
            try: model(**enc, logits_to_keep=1)
            except TypeError: model(**enc)
    finally:
        for h in hooks: h.remove()
    return got


def stopsignal(a):
    """Probability of </think> (closing the thinking) at 40 evenly spaced points of each saved answer."""
    tok, model = load(a.model); sid = end_id(tok)
    rows = json.load(open(a.run))[:a.n]
    head = model.get_output_embeddings()
    norm = None
    for path in ("model.norm", "model.language_model.norm"):
        obj = model
        try:
            for part in path.split("."): obj = getattr(obj, part)
            norm = obj; break
        except AttributeError: pass
    bins = [[] for _ in range(10)]
    for r in rows:
        p_len = tok(r["prompt"], return_tensors="pt")["input_ids"].shape[1]
        enc = tok(r["prompt"] + r["output"], return_tensors="pt").to(model.device)
        got = {}
        h = norm.register_forward_hook(lambda m, i, o: got.__setitem__("h", o.detach()[0]))
        with torch.no_grad():
            try: model(**enc, logits_to_keep=1)
            except TypeError: model(**enc)
            h.remove()
            hid = got["h"][p_len:]
            idx = torch.linspace(0, len(hid) - 1, 40).long()
            prob = torch.softmax(head(hid[idx]).float(), -1)[:, sid].cpu()
        for k, pval in enumerate(prob.tolist()): bins[k * 10 // 40].append(pval)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"chance of closing the thinking (</think>), start -> end of the answer, mean of {len(rows)} answers:")
    print("  " + "  ".join(f"{100 * sum(b) / len(b):5.2f}%" for b in bins))


def easy(a):
    tok, model = load(a.model); sid = end_id(tok)
    rows = gen(tok, model, prompts(tok, a.n, "easy.json"), a.max_tokens, a.batch, sid)
    json.dump(rows, open(RUNS / "observe-easy.json", "w"), indent=1)
    summary(rows, "easy questions")


def finishdir(a):
    tok, model = load(a.model); sid = end_id(tok)
    easy_rows = [r for r in json.load(open(RUNS / "observe-easy.json")) if r["closed_thinking"]]
    hard_rows = [r for r in json.load(open(a.hard)) if r["ran_out"]][:a.n]
    if not easy_rows: raise SystemExit("No easy answer closed its thinking. Run: python diag.py easy --model ...")
    pos, neg = {}, {}
    for r in easy_rows:
        text = r["output"]; cut = text.index("</think>")
        enc = tok(r["prompt"] + text[:cut], return_tensors="pt").to(model.device)
        for L, h in all_layers(model, enc).items(): pos.setdefault(L, []).append(h[-16:].mean(0).float().cpu())   # just before it stops
    for r in hard_rows:
        enc = tok(r["prompt"] + r["output"], return_tensors="pt").to(model.device)
        for L, h in all_layers(model, enc).items(): neg.setdefault(L, []).append(h[-256:].mean(0).float().cpu())  # deep in the loop
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    dirs, scale = {}, {}
    for L in pos:
        d = torch.stack(pos[L]).mean(0) - torch.stack(neg[L]).mean(0)
        dirs[L] = d / d.norm(); scale[L] = torch.stack(neg[L]).norm(dim=1).median().item()
    torch.save({"dirs": dirs, "scale": scale}, RUNS / "finish.pt")
    print(f"finish directions for {len(dirs)} layers, from {len(easy_rows)} finished vs {len(hard_rows)} looping answers")
    print("saved runs/finish.pt")


def steer(a):
    tok, model = load(a.model); sid = end_id(tok)
    f = torch.load(RUNS / "finish.pt")
    rows_all = []
    for L in a.layer:
        u = f["dirs"][L].to(model.device); s = f["scale"][L] * a.alpha
        hook = layers_of(model)[L].register_forward_hook(
            lambda m, i, o: ((o[0] + s * u,) + tuple(o[1:])) if isinstance(o, tuple) else o + s * u)
        t0 = time.time()
        rows = gen(tok, model, prompts(tok, a.n), a.max_tokens, a.batch, sid)
        hook.remove()
        summary(rows, f"steer layer {L} · alpha {a.alpha} · {time.time() - t0:.0f}s")
        for r in rows: r["layer"] = L
        rows_all += rows
    json.dump(rows_all, open(RUNS / f"steer-a{a.alpha}.json", "w"), indent=1)


def bake(a):
    """Weight edit: in layer L's writers, scale up the part of each column that lies along the finish direction.
    W' = W + beta * u (u^T W). Then save the model, and check it with: python overthinker.py observe --model ./qwen-finish"""
    tok, model = load(a.model)
    f = torch.load(RUNS / "finish.pt"); u = f["dirs"][a.layer].to(model.device)
    total = 0
    for name, lin in writers(layers_of(model)[a.layer], u.numel(), "all"):
        W = lin.weight.data
        W += a.beta * torch.outer(u, u @ W)
        total += W.numel()
        print(f"layer {a.layer} {name}: boosted the finish direction in {W.shape[1]} columns")
    out = HERE / a.out; model.save_pretrained(out); tok.save_pretrained(out)
    print(f"edited {total:,} weights · saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("stopsignal"); x.add_argument("--model", required=True); x.add_argument("--run", required=True); x.add_argument("--n", type=int, default=8)
    x = sp.add_parser("easy"); x.add_argument("--model", required=True); x.add_argument("--n", type=int, default=16)
    x.add_argument("--max-tokens", type=int, default=2048); x.add_argument("--batch", type=int, default=16)
    x = sp.add_parser("finishdir"); x.add_argument("--model", required=True); x.add_argument("--hard", default=str(RUNS / "observe-qwen3.5-0.8b.json"))
    x.add_argument("--n", type=int, default=8)
    x = sp.add_parser("steer"); x.add_argument("--model", required=True); x.add_argument("--layer", type=int, nargs="+", default=[15])
    x.add_argument("--alpha", type=float, default=0.5); x.add_argument("--n", type=int, default=8)
    x.add_argument("--max-tokens", type=int, default=4096); x.add_argument("--batch", type=int, default=8)
    x = sp.add_parser("bake"); x.add_argument("--model", required=True); x.add_argument("--layer", type=int, required=True)
    x.add_argument("--beta", type=float, default=1.0); x.add_argument("--out", default="qwen-finish")
    a = ap.parse_args()
    {"stopsignal": stopsignal, "easy": easy, "finishdir": finishdir, "steer": steer, "bake": bake}[a.cmd](a)
