# hierophant

![The Hierophant](hierophant.jpeg)

Explanations of machine-learning models, written by an LLM and **checked against the
model**. Every sentence carries a claim ("for this row, raising `bmi` raised the
prediction"). A verifier tests that claim against the model before the sentence is
kept, so you only read claims that held up.

```python
import hierophant as hp

ds = hp.datasets.load("diabetes")  # data + a fitted GBDT
h = hp.Hierophant.from_dataset(ds, llm="ollama:qwen3:8b")

report = h.ask("Why is this patient's predicted progression low?", row=0)
print(report)  # only the sentences whose claims checked out
print(report.details())  # every claim, its verdict, and the evidence
```

Bring your own model with `hp.Hierophant(model, X, target="...")`. Any scikit-learn-style
regressor or classifier works, and tree models get exact TreeSHAP.

## LLM backends

Local first. Any OpenAI-compatible server, Claude, or your own object with a
`complete(messages)` method:

| spec | where |
|---|---|
| `ollama:qwen3:8b` | local ollama |
| `llamacpp:any` | local llama.cpp server |
| `lmstudio:<model>` | local LM Studio |
| `openai:<model>` | OpenAI (`OPENAI_API_KEY`) |
| `openrouter:<model>` | OpenRouter (`OPENROUTER_API_KEY`) |
| `anthropic:<model>` | Claude (`ANTHROPIC_API_KEY`) |

`hp.Recorder` saves a run's responses, and `hp.Replay` plays them back with no model.

## Datasets

`hp.datasets.names()` lists the shelf: diabetes, breast cancer and wine (bundled with
scikit-learn), plus California housing, Adult income, Bike Sharing and German credit
(downloaded once and cached in `~/.cache/hierophant`). Source and license for each
are in `hp.datasets.REGISTRY`.

## Status

A rewrite, in progress. The 2023 version (SHAP and LIME pasted into a LangChain prompt)
is tagged `legacy-2023`. The lab notebook lives at
[hyperprior.institute/labs/hierophant](https://hyperprior.institute/labs/hierophant/).

## Development

```sh
uv sync
uv run pytest              # offline tests
uv run pytest -m network   # also the downloaded datasets
uv run ruff check . && uv run ruff format --check .
```

MIT licensed.
