# State-Tracking Data Generation

This package generates **State-Tracking** dialogues for TriggerBench. State-Tracking evaluates whether an LLM can detect an implicit state change buried in conversation and proactively intervene when a later action creates a latent risk — without ever being explicitly asked to remind the user.

---

## Seed data

**17 human-authored blueprints** cover all 9 domain × risk-type cells used as few-shot exemplars when scaling with an LLM.

| Domain | Risk types | Seeds |
|--------|------------|-------|
| Transit/Travel | Object_Loss, Task_Failure | seed_01–04 |
| Work/Professional | Object_Forgetting, Privacy_Leak_or_Security, Task_Failure | seed_05–10 |
| Home/Domestic | Object_Damage_or_Forgetting, Safety_Hazard | seed_11–14 |
| Maintenance/DIY | Property_Damage, Security_Breach | seed_15–17 |

`user_dialogues.py` contains only **seed_01**. It is the in-context example for Stage 2. Remaining seeds are stretched from blueprint text alone.

---

## Pipeline

Every scenario follows \( C $\xrightarrow{\Theta}$ T $\to$ A \): a latent **Constraint**, orthogonal thematic turns \($\Theta$\), a **Trigger**, and a required **Action** (or silence on the negative variant).

### Stage 1 — Blueprint design (`generate_blueprints.py`)

An LLM writes new blueprints on a 9-cell combinatorial matrix. 

Each blueprint stores: background, implicit constraint, trigger, time gap (≥ 30 minutes), hidden vulnerability, `distracting_task` (for Positive-Overloaded), `resolution_for_negative` (for Negative-Clean), and gold labels (intent + slot aliases).


### Stage 2 — User-dialogue stretching (`generate_user_dialogues.py`)

Each blueprint becomes a **user-only** dialogue:

```
[Thematic] → [Constraint] → [Thematic] → [Trigger]
```

Thematic turns are topically coherent with the background but orthogonal to the latent risk, and should be answerable from parametric knowledge (no calendar / email / live APIs). The constraint–trigger timestamp gap should match the blueprint (with small tolerance).

### Stage 3 — Dialogue completion / Positive-Clean (`complete_dialogues.py`)

Assistant replies are generated for every user turn **except the trigger**.

### Stage 4 — Positive-Overloaded (`derive_overloaded_variant.py`)

The clean trigger is kept as an exact prefix. An LLM appends a distracting request from `variations.distracting_task`. 

### Stage 5 — Negative-Clean (`derive_negative_variant.py`)

A resolution event is inserted between constraint and trigger. At trigger time the risk is already reminded or resolved, so the correct behavior is to stay silent to avoid over-disturbance.

