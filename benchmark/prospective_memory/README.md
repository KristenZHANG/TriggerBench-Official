# TriggerBench: Prospective Memory Evaluation

This folder runs **prospective memory (PM)** tasks: the model must
spontaneously intervene at a later trigger, without an explicit “remind
me” query.

| Path | Role |
|------|------|
| [`data/`](data/README.md) |PM evaluation data jsonl. Data layout and schema are documented there. |
| [`source/`](source/) | Official PM eval pipelines (blended, RAG, memory methods). |


## Methods

| Method | What it evaluates | Script |
|--------|-------------------|--------|
| **Blended (closed)** | Full dialogue in a chat API (GPT-5.2, GPT-4o, GPT-4.1, …) | `source/run_blended_close.sh` |
| **Blended (open)** | Full dialogue against a local vLLM server | `source/run_blended_open.sh` |
| **RAG** | Embedding retrieval + generation | `source/run_rag.sh` |
| **Mem0** | Mem0-style memory extraction + retrieval, implemented based on [mem0ai](https://github.com/mem0ai/mem0) | `source/run_mem0.sh` |
| **Letta-Sim** | Two-tier core + archival memory, simplified from [Letta/MemGPT](https://github.com/letta-ai/letta) | `source/run_letta_sim.sh` |
| **A-MEM** | Agentic memory with LLM-driven evolution, uses [Memory-in-the-LLM-Era](../Memory-in-the-LLM-Era) | `source/run_amem.sh` |

All methods share `pm_scorer.py` and `pm_judge.py`. Generation
temperature defaults to **0.6** (paper); API inference defaults to
**gpt-4.1**; Judge is always **GPT-4o**
at temperature **0.0**.

## Metrics

- **Slot Match** — case-insensitive rule-based recall of required constraint
  entities. Positive tasks only; N/A on negatives.
- **PM Acc (LLM-as-a-Judge)** — official accuracy. GPT-4o at
  **temperature 0**, as in the paper. The judge is an *intent detector*:
  YES = the reply conveys the reference proactive intent (paraphrase OK;
  keyword-only mentions are NO).

On **negative** tasks the same positive reference intent is used.
Judge **YES** = false alarm (score 0); **NO** = correct silence (score 1).
Logical Adherence negatives use intent-only judging (no flipped
constraint/trigger in the prompt); other negatives use full context.


## Setup

Recommend Python ≥ 3.10. Eval packages are listed in [`requirements.txt`](requirements.txt)
(closed/open blended, RAG, Letta-Sim, judge). vLLM is separate, then point `run_blended_open.sh` at it.
Mem0 / A-MEM need extra packages; see comments in `requirements.txt`.

```bash
pip install -r requirements.txt
cd source
cp .env.example .env   # fill in API keys
```

**Fill in `.env` before running.** The **judge is always GPT-4o** (`pm_judge.py`), independent of `--deployment`. RAG / Mem0 / Letta-Sim / A-MEM also need the embedding fields.

### Shared flags

| Flag | Meaning |
|------|---------|
| *(none)* | All categories × variants that exist in the data |
| `--category state_tracking` | One dimension (`temporal_grounding`, `logical_adherence`, `attention_recovery`, `safe_coding`) |
| `--variant positive_clean` | `positive_clean` / `positive_overloaded` / `negative_clean` |
| `--bin 40K` | Default: base. Long context version is only supported by blended & RAG |
| `--deployment gpt-4.1` | Chat model for inference (closed / RAG / memory). Judge stays GPT-4o |

---

## Blended (closed-source)

```bash
bash run_blended_close.sh
bash run_blended_close.sh --reasoning --deployment gpt-5.2   # reasoning APIs
```

## Blended (open-source) — deploy first

```bash
bash run_vllm_server.sh                  # default: Qwen3-32B
# bash run_vllm_server.sh instruct       # Qwen3-235B-A22B-Instruct-2507-FP8
# bash run_vllm_server.sh thinking       # Qwen3-235B-A22B-Thinking-2507-FP8
# bash run_vllm_server.sh gemma3-27b     # gemma-3-27b-it

# then: 
bash run_blended_open.sh --model <served-model-name>
```

## RAG / Mem0 / Letta-Sim / A-MEM

```bash
# fix concurrency at 1
bash run_rag.sh --top-k 6
bash run_mem0.sh          
bash run_letta_sim.sh
bash run_amem.sh
```

A-MEM imports `../Memory-in-the-LLM-Era/code/Method/amem/memory_layer.py`
(algorithm only). Mem0 needs `mem0ai`.
