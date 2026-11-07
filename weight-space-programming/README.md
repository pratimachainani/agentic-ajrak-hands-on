# Weight-space programming: fixing the overthinker

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pratimachainani/agentic-ajrak-hands-on/blob/main/weight-space-programming/overthinker-colab.ipynb)

A small, real reproduction of the idea in "Sparks of Weight-Space Programming" (Omar Khattab's group, MIT CSAIL): change a model's behaviour by editing its weights, with no training. Explainer: [agenticajrak.com/glossary/weight-space-programming](https://agenticajrak.com/glossary/weight-space-programming.html).

The model is Qwen3.5-0.8B. On hard maths problems it overthinks: it runs out of tokens instead of answering.

## Files
| File | What it is |
|---|---|
| `overthinker-colab.ipynb` | The whole experiment in Google Colab (free T4 GPU). Start here. |
| `overthinker.py` | `observe` (count answers that run out of tokens), `direction` (find the "stuck" direction in the residual stream), `patch` (remove it from the weights), `score` |
| `diag.py` | Diagnosis: `stopsignal` (chance of closing the thinking), `easy`, `finishdir` (a "finish" direction per layer), `steer` (test it at runtime with a hook), `bake` (write it into the weights) |
| `problems.json` | 32 practice maths problems, written for this test |
| `easy.json` | 16 easy questions, written for this test |

## Run it
**Colab:** click the badge, set Runtime → Change runtime type → T4 GPU, run the cells top to bottom.

**Mac (Apple silicon, 16 GB+):**
```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch transformers accelerate modelscope
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ./qwen3.5-0.8b
python overthinker.py observe --model ./qwen3.5-0.8b --max-tokens 8192
```
Everything stays in this folder (about 6–8 GB). No `sudo`, and no downloaded code is run (`trust_remote_code` is off).

## What we found (6 Nov 2026)
| Try | Edit | Result |
|---|---|---|
| Before | none | 31 / 32 ran out of 8,192 tokens; 31 / 32 repeating themselves |
| 1 | remove the "stuck" direction from the top 64 MLP output columns, layers 15 + 19 (131,072 weights) | 32 / 32 ran out |
| 2 | same direction, every column of layers 15 + 19 (7.3M weights) | 8 / 8 ran out |
| 3 | every layer, every matrix that writes to the residual stream (138M weights) | 32 / 32 ran out, but repeating fell to 13 / 32: it stopped repeating and started second-guessing |
| Diagnose | chance of closing its thinking | 0.00% across the answers |
| Diagnose | easy questions | closed their thinking 8 / 16, answered 16 / 16 |
| Steer (runtime) | push a "finish" direction at layer 18 | closed thinking 1 / 8 → 6 / 8 at strength 1.0; 0.25 did nothing; 2.0 broke the model |
| Bake | boost that direction in layer 18's weights | 0 / 8 |

The "finish" signal is real and sits late in the model; repetition lives in the middle layers. Turning the finish direction into a clean weight edit is the open step.
Results vary a little between runs (answers are sampled at temperature 0.6).
