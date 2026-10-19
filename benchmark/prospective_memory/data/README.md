# TriggerBench Evaluation Data

Official **prospective memory (PM)** set at two lengths (Base and 40K). The main analysis in the paper are conducted on the **Base** context.


## Overall design 

Every item uses the same skeleton:

\[
C $\xrightarrow{\Theta}$ T $\to$ A
\]

| | Role |
|--|------|
| \(C\) | Constraint: an ordinary utterance that plants a latent risk or policy. |
| \($\Theta$\) | Orthogonal thematic (or external injected) turns. No gold leakage. |
| \(T\) | Trigger: benign in isolation; with \(C\), the moment to act. Always the last user turn. |
| \(A\) | Positive: a proactive intervention. Negative: silence (not remind). |


### Contrastive variants

| Variant | Purpose |
|---------|---------|
| **Positive-Clean** | Constraint active; trigger without a concurrent request. Intervene. |
| **Positive-Overloaded** | Same prefix as Pos-Clean (**prefix preservation**). Only \(T\) grows: clean trigger text plus a distracting ask. Evaluate attention hijacking. Gold intent/slots unchanged. |
| **Negative-Clean** | False-alarm control. Construction differs by dimension (below). Gold `intent.summary` remains the **positive** reminder intentionally. |

Temporal Grounding and Attention Recovery have no Neg-Clean.

### Context lengths

| Split | Folder | Construction |
|-------|--------|----------------|
| Base | `base_context/` | Paper “Base Context” (~2.5K tokens), main analysis target |
| 40K | `40K_context/` | Same gold \(C\)/\(T\); filler inserted before \(T\) (recipe is dimension-specific). |

```
data/
├── base_context/{positive_clean,positive_overloaded,negative_clean}/<dim>/<dim>_base.jsonl
└── 40K_context/…  same tree, filenames *_40K.jsonl
```

### Record (all dimensions)

```text
{
  "task_id", "category", "variant", "bin",   // bin is "base" | "40K" 
  "sample": {
    "meta": { ... },                         // taxonomy; Base only — join 40K via task_id
    "turns": [ { "idx", "role", "text", "timestamp", "function" } ],
    "ground_truth": {
      "intent":  { "summary", "constraint" },
      "trigger": { "description", "trigger_turn_idx" },
      "action":  { "required_slots", "slot_values" }
    }
  }
}
```

`intent.summary` is the judge’s reference intent (still the positive reminder
on Neg-Clean). `trigger_turn_idx` locates \(T\) in `turns` (always last);
official eval reads that utterance.

### 1,265 PM tasks

| Dimension | Folder | Cue | \(n\) |
|-----------|--------|-----|-------|
| State-Tracking | `state_tracking/` | Implicit state that later becomes risky | 108 × 3 = 324 |
| Temporal Grounding | `temporal_grounding/` | Point-in-time commitment | 143 × 2 = 286 |
| Logical Adherence | `logical_adherence/` | User-stated conditional policy | 133 × 3 = 399 |
| Attention Recovery | `attention_recovery/` | Suspended task after a cognitive hijack |  56 × 2 = 112 |
| Safe Coding | `safe_coding/` | Temporary debugging / safety shortcut | 48 × 3 = 144 |

### State-Tracking

Ephemeral state mentioned in passing (e.g. an item left in a trunk), then a
later action that makes that state dangerous. \(C\) is narration; \(T\) is a
state change (“I’m getting out”).

Neg-Clean uses **risk resolution**: an intermediate turn undoes the risk
before the *same* trigger sentence through explicit remind or resolve. 

40K: [MultiWOZ](https://huggingface.co/datasets/pfb30/multi_woz_v22)(Budzianowski et al., 2018) filler before \(T\).

### Temporal Grounding

Keep a timed commitment (absolute or relative) and remind when a later
status update means the deadline is about to be missed. 

40K: [MultiWOZ](https://huggingface.co/datasets/pfb30/multi_woz_v22)(Budzianowski et al., 2018) filler before \(T\).

### Logical Adherence

Encode a user-stated policy and test it at \(T\). Unlike State-Tracking,
Neg-Clean is a **minimal pair** (one flipped unit), not an inserted
resolution turn. Three logic types:

| Logic type | Hops | Positive | Negative (what flips) | 
|------------|------|----------|------------------------|
| If-Then (48) | 2: rule then condition | Event A occurred → remind Action C at \(T\) | **Condition polarity** (A did not occur), plus the following assistant reply | 
| Mutual Exclusion (40) | 1: rule only | \(T\) **violates** the red line / threshold | **Trigger** (comply instead of violate) | 
| Only-After (45) | 1: rule only | \(T\) announces the prerequisite is **complete** | **Trigger** (look-alike / not yet done) | 


Neg-Clean **nulls** gold `intent.constraint` and `trigger.description`
(the flipped sentences stay in `sample.turns`). The official judge is
**intent-only** for this cell
(`is_negative and category == "logical_adherence"`), so a flipped
\(C\)/\(T\) is never shown beside the still-positive reference intent
to avoid conflict.

40K: [MultiWOZ](https://huggingface.co/datasets/pfb30/multi_woz_v22)
(Budzianowski et al., 2018) filler before \(T\).

### Attention Recovery

A casually mentioned in-progress Task A, then an orthogonal hijack
(math or code). When the interruption is declared over, remind the user
to resume Task A. 

40K: Extra filler is further [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)(Guha et al., 2026) math/code
pairs, same family
as the item. Length comes from long chain-of-thought replies, not from
thousands of short turns.

### Safe Coding

Professional workflow: a developer shortcut planted at \(C\), then a
deploy / merge / expose request at \(T\). The middle is
[Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868) agent trajectory, not LLM-written small talk.

Neg-Clean is **risk resolution** (shortcut undone before the same \(T\)),
with the full judge — same protocol as State-Tracking, not Logical
Adherence.

40K: replaces the Base orthogonal connection trajectory with longer items from [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868).

## License

The TriggerBench-authored portion of this set (constraints, triggers, gold, and
construction) is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
40K / Safe Coding dialogues include third-party excerpts (MultiWOZ, OpenThoughts,
Terminal-Bench trajectories); see the repository [`NOTICE.md`](../../../NOTICE.md).

## Note on the 40K split

We first generate the Base dialogues (natural timing, event-driven gaps), then stretch them to 40K by inserting external filler (or, for Safe Coding, swapping in a longer trajectory). The 40K conversations are therefore more mechanical than Base: filler is packed into a late gap whose timestamps are already fixed, so many consecutive turns may share the same clock time in order to keep the sequence non-decreasing. We apply rule-based filters to drop filler that mentions dates or times and to avoid gold-slot leakage, but this cannot be guaranteed to be complete. Our main experiments and analyses in the paper thus use the Base version.
