# Attention Recovery Data Generation

Attention Recovery evaluates whether an LLM can encode a casually mentioned **in-progress Task A**, survive a long, semantically coherent cognitive hijack (Task B), and **proactively remind the user to resume Task A** once the interruption is declared over — without ever being explicitly asked to remember.

The interruption is an unrelated math/code problem (Task B). The agent should remind after the interruption resolves. 
This package stops at **placeholder templates**, which can be filled with external math/code problem into `<ASSISTANT_PAYLOAD_PLACEHOLDER>`. We use math and code data from [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k).

---

## Dual-task design

Difficulty is **attentional**, not logical or temporal. Task A is mentioned once as the constraint, as a brief in-progress status, then quickly displaced by thousands of tokens of genuine reasoning:

```
Task A suspended  --> Task B reasoning-->  Task B resolved  →  remind Task A
```

Two architectural constraints keep the evaluation valid.

**Orthogonality.** Task A must not overlap the payload domains of Task B (e.g., math, code) to avoid retrieval cue leakage.

**Template–payload decoupling.** Stages 1–4 do not write the actual math/code problem. The interruption user turn holds only a **bridge prefix**; the assistant turn is the static token `<ASSISTANT_PAYLOAD_PLACEHOLDER>`. 

---

## Seed data

**8 human-authored blueprints** cover the orthogonal 4 domain × 2 payload type (one seed per cell). 

| Seed | Domain | Payload |
|------|--------|---------|
| `interrupt_seed_001` | Work/Professional | math |
| `interrupt_seed_002` | Work/Professional | code |
| `interrupt_seed_003` | Home/Domestic | math |
| `interrupt_seed_004` | Home/Domestic | code |
| `interrupt_seed_005` | Finance/Commerce | math |
| `interrupt_seed_006` | Finance/Commerce | code |
| `interrupt_seed_007` | Study/Academic | math |
| `interrupt_seed_008` | Study/Academic | code |

---

## Time gaps

LLMs are unreliable at sampling realistic interruption timing. Stage 1 therefore **samples both gaps in Python** and injects them as hard constraints. The LLM only writes scenario text.

| Gap | Range | 
|-----|-------|
| Task A → interruption | 2–6 minutes | 
| Interruption → trigger | 40–90 minutes | 

Bridge prefixes are also **not** written by the LLM. Each payload type has three relationship/urgency variants; Stage 1 samples one and injects the exact string.

The injected prefix includes a short **pause-and-resume** sentence (“Let me put aside what I was doing — I’ll get back to it once we’re done…”). It does **not** name Task A. 

---

## Combinatorial matrix

Blueprints are generated on two orthogonal axes.

**Axis 1 — 4 domains**

| Domain | Archetype | 
|--------|-----------|
| Work/Professional | Performance reviews, slide decks, admin paperwork | 
| Home/Domestic | Seasonal chores, furniture assembly |
| Finance/Commerce | Statement reconciliation, insurance comparison | 
| Study/Academic | Humanities essays and reading | 

**Axis 2 — payload type**

| Type | Bridge framing | Resolution cue |
|------|----------------|----------------|
| Math | A relative needs help with a competition problem | “explained the math problem…” |
| Code | A colleague / friend hits a coding issue | “solved that coding problem…” |

