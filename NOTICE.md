# Notices

This file records licenses that apply in addition to the top-level MIT
[`LICENSE`](LICENSE) (TriggerBench original code) and CC BY 4.0 for the
TriggerBench evaluation set.

## TriggerBench original work

- **Code** (evaluation runners, judges, data-generation pipeline, and
  original wrappers such as `benchmark/prospective_memory/source/`): MIT.
  See [`LICENSE`](LICENSE).
- **Evaluation data** (`benchmark/prospective_memory/data/`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Please cite [TriggerBench](https://arxiv.org/abs/2606.23459) if you use it.

Some 40K / Safe Coding dialogues **embed third-party text**. Those excerpts
keep their original licenses below; CC BY 4.0 covers the TriggerBench-authored
constraint, trigger, gold, and surrounding construction, not a relicensing of
the upstream corpora.

## Vendored A-MEM (Memory-in-the-LLM-Era)

[`benchmark/Memory-in-the-LLM-Era/`](benchmark/Memory-in-the-LLM-Era/) is a
**partial copy** of [Memory in the LLM Era](https://github.com/Yanchen398/Memory-in-the-LLM-Era)
(Wu et al.; [arXiv:2604.01707](https://arxiv.org/abs/2604.01707)), limited to
the A-MEM implementation under `code/Method/amem/`.

Upstream license: MIT, Copyright (c) 2026 Yanchen Wu.
The license text is copied at
[`benchmark/Memory-in-the-LLM-Era/LICENSE`](benchmark/Memory-in-the-LLM-Era/LICENSE).

A-MEM itself was introduced by Xu et al.,
[Agentic Memory](https://github.com/WujiangXu/AgenticMemory)
([arXiv:2502.12110](https://arxiv.org/abs/2502.12110)). TriggerBench uses the
Memory-in-the-LLM-Era reimplementation, not that original repository.

## Third-party sources used as filler / trajectories

These datasets appear as orthogonal context (or Safe Coding middle turns),
not as the TriggerBench gold constraint/trigger:

| Source | Use in TriggerBench | License (upstream) |
|--------|---------------------|--------------------|
| [MultiWOZ 2.2](https://huggingface.co/datasets/pfb30/multi_woz_v22) (Budzianowski et al., 2018) | 40K filler for State-Tracking, Temporal Grounding, Logical Adherence | Apache-2.0 |
| [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) (Guha et al., 2025) | Attention Recovery hijack / 40K math–code filler | Apache-2.0 |
| [Terminal-Bench trajectories](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories) ([Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868)) | Safe Coding in-between agent traces (Base and 40K) | Apache-2.0 |

Apache-2.0 material is included under that license. A copy is at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). This NOTICE and the
upstream project names/URLs are the required attribution. 

## Independent reimplementations (not vendored)

The following evaluation methods in `benchmark/prospective_memory/source/`
are TriggerBench wrappers / simplified reimplementations. They are MIT as
TriggerBench code, but the **ideas** come from those projects; we do not
ship their full upstream trees:

- [Mem0](https://github.com/mem0ai/mem0)
- [Letta / MemGPT](https://github.com/letta-ai/letta)

Use of those names does not change their respective upstream licenses if you
install the original packages separately.
