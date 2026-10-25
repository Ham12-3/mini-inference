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
| 1 | Naive baseline (no cache) | done |
| 2 | Own Qwen2 model + single-sequence KV cache | done |
| 3 | Slot-based KV cache + static batching | done |
| 4 | Scheduler + continuous batching | |
| 5 | Benchmarks vs baselines and vLLM | |
| 6 | Extras: streaming API, paged KV cache | |

## Results so far

![Single-request decode speed](benchmarks/results/chart1_single_request.png)

Single request, Qwen2.5-0.5B-Instruct, fp16, greedy, Tesla T4 (Colab):

| Output tokens | Naive (no cache) | HF `generate()` | My engine (KV cache) |
|---:|---:|---:|---:|
| 64 | 30.1 tok/s | 33.0 tok/s | 45.2 tok/s |
| 256 | 30.0 tok/s | 33.3 tok/s | 43.8 tok/s |
| 1024 | 12.1 tok/s | 33.7 tok/s | 44.7 tok/s |

Parity: my model's float32 logits are bit-identical to Hugging Face's on CPU
and within 1e-3 on the GPU. In float16, greedy output matches HF `generate()`
for 50 tokens on 4 of 5 prompts; the fifth matches for 40 tokens and then
hits an exact fp16 tie between the top two tokens (see
`tests/test_model_parity.py` for how ties are verified).

### Static batching (Stage 3)

![Throughput vs batch size](benchmarks/results/chart2_static_batching.png)

Each request generates 128 tokens; all requests in a batch start together.

| Batch size | My engine | HF `generate()` |
|---:|---:|---:|
| 1 | 40 tok/s | 34 tok/s |
| 8 | 291 tok/s | 281 tok/s |
| 32 | 987 tok/s | 1,076 tok/s |

Going from 1 to 32 requests per batch makes each batch only 28% slower
(3.24 s to 4.15 s), so throughput grows 25x: a decode step reads the weights
once whether it serves one sequence or 32. Batched greedy outputs are
identical to running each prompt alone (fp32 and fp16, `tests/test_kv_cache.py`).
At batch 32 HF is about 9% faster: my decode step copies each sequence's
keys/values out of its cache slot and repeats them for every query head, and
that work grows with the batch.

## Run it

On Colab: open [`notebooks/run_on_colab.ipynb`](https://colab.research.google.com/github/Ham12-3/mini-inference/blob/master/notebooks/run_on_colab.ipynb),
pick a T4 GPU runtime and choose **Runtime → Run all**.

Locally:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
python -m benchmarks.env_info
```
