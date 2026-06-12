# InfraHeal AI — Cloud Dependency Fixes

## Environment: JupyterLab (ROCm + vLLM, Python 3.12)

### Final working fix (run this once at start)

```bash
pip install "transformers>=4.55.2,<5.0" "numpy<2.3"
```

Then:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9 --max-model-len 8192
```

### Known compatible versions (verified)

| Package | Allowed Range | Constraint From | Why |
|---|---|---|---|
| `transformers` | `>=4.55.2,<5.0` | vLLM 0.11.0rc2 | needs `AutoVideoProcessor` (4.49+) |
| `numpy` | `<2.3` (i.e. 2.2.x) | numba 0.61.2 | numba only supports ≤2.2 |
| `huggingface-hub` | `>=1.2.0,<2.0` | gradio 6.18.0 | cloud default is 1.19.0 — works |
| `opencv-python-headless` | `numpy<2.3.0,>=2` | opencv 4.12.0 | satisfied by numpy 2.2.x |

### Error → Root Cause → Fix cheat sheet

| Error | Root Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'AutoVideoProcessor'` | transformers < 4.49 | `pip install "transformers>=4.55.2,<5.0"` |
| `Numba needs NumPy 2.2 or less. Got NumPy 2.4` | numpy too new from transformers 5.x cascade | `pip install "numpy<2.3"` |
| `huggingface-hub>=0.34.0,<1.0 is required, but found 1.19.0` | transformers 4.x needs hub < 1.0 | Use transformers 5.x OR pin hub to 0.36.x |
| `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers ≥ 4.50 removed this | `pip install "transformers>=4.55.2,<5.0"` |
| `ImportError: cannot import name 'BatchEncoding'` | transformers + hub mismatch | `pip install "transformers>=4.55.2,<5.0" "huggingface-hub>=1.2.0,<2.0"` |

### DON'T run these (proven trouble)

- `pip install transformers==4.48.3` — too old, missing `AutoVideoProcessor`
- `pip install "transformers>=4.55.2"` (no `<5.0` pin) — pulls 5.x → upgrades numpy → breaks numba
- `pip install --force-reinstall` on transformers — cascading numpy/opencv/numba breakage
