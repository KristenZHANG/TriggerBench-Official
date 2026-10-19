# Logical Adherence Data Generation

Logical Adherence evaluates whether an LLM can encode a user-stated conditional rule, judge whether that rule’s condition is later met or violated, and intervene—or stay silent.

## Three logic types

Logical Adherence tests **user-defined policies**. Every blueprint is a minimal pair: a positive case where the agent must act, and a negative case where it must stay silent.

### Forward If-Then (multi-hop)

```
If [Event A] happens → I should do [Action C] at [Time/Event B]
```

The agent must remember the If-Then rule `rule_constraint` turn, observe whether Event A occurred in a later `condition_constraint` turn, and decide at trigger time whether to remind the user of Action C. The negative case is the same trigger after Event A did **not** occur.

### Mutual Exclusion (red-line / threshold)

```
I must NOT do [Action X] / [Action X] must not exceed [Threshold Y]
```

The agent evaluates a later request against an absolute constraint. The positive trigger **violates** the rule; the negative trigger **complies**. The distinction may be quantitative ($85 vs $25 against a $50 limit) or qualitative (alcohol vs a virgin mojito while on antibiotics).

### (Only)-After (prerequisite unlock)

```
I will do [Action X] after [Event Z] finishes
```

The agent must recognize exact completion of a prerequisite—not a partial or intermediate state. The positive trigger announces full completion; the negative trigger describes a look-alike that is not yet done.


## Minimal-pair positive / negative

Within each blueprint, positive and negative dialogues differ by a single semantic unit:

| Logic type | What differs | What stays identical |
|------------|--------------|----------------------|
| Forward If-Then | `condition_constraint` polarity | Rule, thematic filler, and trigger |
| Mutual Exclusion | Trigger text (violate vs comply) | Entire prior conversation |
| Only-After | Trigger text (complete vs pending) | Entire prior conversation |

An agent that always reminds fails half the cases. Success requires both sensitivity and specificity.


## Combinatorial matrix and variants

Blueprints sit on **7 domains × 3 logic types**. Production generation uses 7 items per cell, plus 6 human-authored seeds.

**Domains:** Work/Professional, Home/Domestic, Finance/Commerce, Health/Dietary, Parenting/Social, Transit/Travel, Study/Academic.

## Implicit rule

The rule is never framed as an explicit request for help. The user states a personal policy in passing—“If the CEO mentions the Q3 project, I should submit the report before I leave”—with **no** “remind me”, “set an alarm”, or similar phrasing. 


## Two dialogue templates

Stage 2 stretches blueprints into **user-only** dialogues. Structure depends on logic type.

**Type A (Forward If-Then) — 6 sections:**

```
[Thematic] → [rule_constraint] → [Thematic]
→ [condition_constraint] → [Thematic] → [trigger]
```

**Type B (Mutual Exclusion, Only-After) — 4 sections:**

```
[Thematic] → [rule_constraint] → [Thematic] → [trigger]
```

## Seed data

**6 human-authored blueprints** cover the three logic types (two each). `user_dialogues.py` contains **3 few-shot dialogues**, one per logic type; the remaining seeds are stretched from blueprint text alone.

| Seed | Domain | Logic type | Few-shot dialogue |
|------|--------|------------|-------------------|
| `seed_ifthen_001` | Work/Professional | Forward_If-Then | yes |
| `seed_ifthen_002` | Home/Domestic | Forward_If-Then | — |
| `seed_exclusion_001` | Finance/Commerce | Mutual_Exclusion | yes |
| `seed_exclusion_002` | Health/Dietary | Mutual_Exclusion | — |
| `seed_onlyafter_001` | Parenting/Social | Only-After | yes |
| `seed_onlyafter_002` | Parenting/Social | Only-After | — |

All scenarios are synthetic. Names, medications, and workplace events are fictional props, not real user or institutional data.
