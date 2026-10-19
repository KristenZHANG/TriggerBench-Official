# Temporal Grounding Data Generation

Temporal Grounding evaluates whether an LLM can encode a casually mentioned **strict point-in-time** commitment, perform temporal arithmetic over the conversation clock, and proactively intervene when a later utterance shows the user is about to miss that exact moment.

---

### Strict point-in-time Design

Every scenario uses **exact-moment** semantics: the target action is valid only at the specified instant. 
This removes the evaluation ambiguity of flexible deadlines, where an agent could “succeed” by reminding at any earlier point. 

---

## Seed data

**4 human-authored blueprints** cover the orthogonal 2 × 2 of time-logic type × trigger pattern. Unlike State-Tracking, **every seed has a few-shot user dialogue** in `user_dialogues.py`; Stage 2 picks the matching cell as the in-context example.

| Seed | Domain | Time logic | Trigger pattern |
|------|--------|------------|-----------------|
| `seed_01_absolute_blind_action` | Social | Absolute_Time | Time_Blind_Action |
| `seed_02_relative_time_query` | Workplace/Professional | Relative_Time | Time_Query |
| `seed_03_absolute_time_query` | Social | Absolute_Time | Time_Query |
| `seed_04_relative_blind_action` | Workplace/Professional | Relative_Time | Time_Blind_Action |

`seed_01` and `seed_03` share the midnight birthday-call pact; `seed_02` and `seed_04` share the lysis-buffer / centrifuge protocol. The pairing is intentional: the same commitment is crossed with both trigger patterns so the 2 × 2 matrix has a quality anchor in every cell.

---

## Time Control

LLMs are unreliable at temporal arithmetic. Stage 1 therefore **pre-computes every timestamp in Python** and injects the values as hard constraints. The LLM only writes scenario text.

| Anchor | Meaning | 
|--------|---------|
| `constraint_talking_time` | When the user mentions the commitment | 
| `duration_time` | Natural-language offset (Relative_Time only) | 
| `exact_target_action_time` | The instant the action must occur | 
| `trigger_talking_time` | When the user utters the trigger | 


- **Absolute_Time**: the constraint text must name the target clock time; `duration_time` is `null`.
- **Relative_Time**: the constraint text must name the duration and must **not** reveal the computed target time.

---

## Combinatorial matrix

Blueprints are generated on three orthogonal axes. Risk type is **not** a forced input: missed-window consequences (social faux pas, ruined sample, missed drop) arise from the scenario itself.

**Axis 1 — 7 domains**

| Domain | Archetype | 
|--------|-----------|
| Social | Midnight pact, anniversary minute | 
| Science_Lab | Lysis / centrifuge / PCR | 
| Home | Temporary door code, appliance cycle |
| Nature | Eclipse, ISS pass, tidal peak | 
| Digital_Entertainment | Sneaker drop, ticket release | 
| Work | Global launch, server switchover | 
| Health | Insulin / dose window |

**Axis 2 — time logic**

| Type | Constraint wording | Agent’s job |
|------|--------------------|-------------|
| Absolute_Time | “The drop goes live at exactly 10:00” | Compare current time to the stated target |
| Relative_Time | “The reaction needs exactly 45 minutes” | Add duration to constraint time |

**Axis 3 — trigger pattern**

| Pattern | User behavior | Agent’s job |
|---------|---------------|-------------|
| Time_Blind_Action | Announces a time-consuming activity | Leaving now suggests missing the instant |
| Time_Query | Asks “what time is it?” while absorbed | Answer the clock **and** warn about the window |

