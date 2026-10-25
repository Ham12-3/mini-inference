# Mini LLM Inference Engine

A small LLM inference engine in Python and PyTorch, in the style of
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm): my own Qwen2 model
implementation, my own KV cache, a request queue and scheduler, and continuous
batching, all running on a single Colab T4 GPU.

> Work in progress. This README will grow into a full write-up (how inference
> works, design decisions, benchmark results) as each stage lands.

## Status

| Stage | What | Status |
|---|---|---|
| 0 | Repo skeleton, Colab notebook, GPU + model check | done |
| 1 | Naive baseline (no cache) | |
| 2 | Own Qwen2 model + single-sequence KV cache | |
| 3 | Slot-based KV cache + static batching | |
| 4 | Scheduler + continuous batching | |
| 5 | Benchmarks vs baselines and vLLM | |
| 6 | Extras: streaming API, paged KV cache | |

## Run it

On Colab: open [`notebooks/run_on_colab.ipynb`](https://colab.research.google.com/github/Ham12-3/mini-inference/blob/master/notebooks/run_on_colab.ipynb),
pick a T4 GPU runtime and choose **Runtime → Run all**.

Locally:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
python -m benchmarks.env_info
```
