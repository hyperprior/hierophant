"""First light: `ask` mode on a local model, scored by the verifier.

    uvx --from mlx-lm mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080
    uv run python examples/first_light.py [--replay]
"""

import sys
import time
from collections import Counter

import hierophant as hp

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
PLAN = {"diabetes": 20, "breast_cancer": 10, "california_housing": 10}
QUESTION = "Why did the model make this prediction for this row?"

replay = "--replay" in sys.argv
totals, tokens, seconds = Counter(), 0, 0.0
for name, n in PLAN.items():
    ds = hp.datasets.load(name)
    path = f"runs/first-light-{name}.jsonl"
    if replay:
        llm = hp.Replay(path)
    else:
        llm = hp.Recorder(hp.OpenAICompat(MODEL, "http://localhost:8080/v1"), path)
    h = hp.Hierophant.from_dataset(ds, llm=llm)
    c, sentences = Counter(), 0
    t0 = time.perf_counter()
    for row in range(n):
        r = h.ask(QUESTION, row=row)
        c.update(s.check.verdict.value for s in r.sentences)
        c["parse_error"] += bool(r.error)
        sentences += len(r.sentences)
        tokens += r.input_tokens + r.output_tokens
    dt = time.perf_counter() - t0
    seconds += dt
    totals += c
    print(f"{name:20s} rows={n:3d} sentences={sentences:3d} "
          f"verified={c['verified'] / max(sentences, 1):.0%} {dict(c)} {dt:.0f}s")

n = sum(v for k, v in totals.items() if k != "parse_error")
print(f"{'all':20s} faithfulness={totals['verified'] / n:.0%} {dict(totals)} "
      f"tokens={tokens} wall={seconds:.0f}s")
