"""PM Judge — LLM-as-a-Judge for Prospective Memory evaluation.

Checks whether the assistant's response semantically conveys the gold
proactive intent. Uses category-specific few-shot examples so the judge
understands the distinct reasoning patterns:

  state:       physical object/location → remind to retrieve/act
  time:        time-sensitive deadline → remind before too late
  conditional: if-then rule / exclusion / only-after → enforce the rule
  interrupt:   suspended task after long distraction → remind to resume

Relaxed criteria: accepts core reminder even without all entity details.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  System Prompt (relaxed version)
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a proactive AI assistant benchmark.

═══ TASK ═══

You will be given:
1. A **Constraint** — what the user originally told the assistant earlier in \
the conversation (the reason a proactive reminder is needed). This is the \
background context that the assistant should have remembered.
2. A **Reference Intent** — a sentence describing what the assistant SHOULD \
proactively communicate (e.g. "Remind the user to grab the cello from the \
trunk before exiting the taxi").
3. The **Trigger Utterance** — the user's message at the moment the reminder \
should fire.
4. The **Assistant's Full Response** at that moment.

Your job: determine whether the assistant's response **proactively conveys** \
the core message of the reference intent, given the constraint context.

═══ CRITERIA ═══

Answer YES if:
- The response communicates the CORE proactive action described in the \
reference intent — it alerts, reminds, or warns the user about the key \
concern.
- The response does NOT need to mention every entity or detail. As long as \
the essential reminder/warning is present and recognizable, it counts.
  Example: intent = "remind user to grab cello from trunk before leaving" \
  → "Don't forget your cello!" is YES (core reminder present). \
  → "Don't forget your things in the trunk!" is also YES.
- Paraphrasing, partial wording, or weaving into a longer reply all count.
- Partial but meaningful reminders count as YES — e.g., mentioning the key \
object/action without specifying exact location, or mentioning the deadline \
without the exact time.

Answer NO if:
- The response does NOT contain any proactive reminder, warning, or \
intervention related to the reference intent.
- Merely mentioning a related keyword in a non-reminder context does NOT \
count (e.g., intent = "remind user to take insulin" → response says \
"Mochi had insulin this morning" with no reminder → NO).
- The response only addresses the trigger utterance with no reference to \
the intent's reminder action at all.

═══ OUTPUT FORMAT ═══

Respond with ONLY a JSON object (no markdown, no extra text):
{"verdict": "YES" or "NO", "reason": "<one-sentence explanation>"}\
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Intent-only System Prompt (for conditional negatives)
# ═══════════════════════════════════════════════════════════════════════════

# Conditional negatives flip constraint (if-then) or trigger (mutual_exclusion,
# only_after), so providing them to the judge causes logical mismatch.
# This prompt evaluates only intent vs response.

_JUDGE_SYSTEM_PROMPT_INTENT_ONLY = """\
You are an expert evaluator for a proactive AI assistant benchmark.

═══ TASK ═══

You will be given:
1. A **Reference Intent** — a sentence describing what the assistant SHOULD \
proactively communicate (e.g. "Remind the user to grab the cello from the \
trunk before exiting the taxi").
2. The **Assistant's Full Response** at that moment.

Your job: determine whether the assistant's response **proactively conveys** \
the core message of the reference intent.

═══ CRITERIA ═══

Answer YES if:
- The response communicates the CORE proactive action described in the \
reference intent — it alerts, reminds, or warns the user about the key \
concern.
- The response does NOT need to mention every entity or detail. As long as \
the essential reminder/warning is present and recognizable, it counts.
  Example: intent = "remind user to grab cello from trunk before leaving" \
  → "Don't forget your cello!" is YES (core reminder present). \
  → "Don't forget your things in the trunk!" is also YES.
- Paraphrasing, partial wording, or weaving into a longer reply all count.
- Partial but meaningful reminders count as YES — e.g., mentioning the key \
object/action without specifying exact location, or mentioning the deadline \
without the exact time.

Answer NO if:
- The response does NOT contain any proactive reminder, warning, or \
intervention related to the reference intent.
- Merely mentioning a related keyword in a non-reminder context does NOT \
count (e.g., intent = "remind user to take insulin" → response says \
"Mochi had insulin this morning" with no reminder → NO).
- The response only addresses the user's request with no reference to \
the intent's reminder action at all.
- The response mentions the topic as a future plan or conditional action \
(e.g., "once X happens, you should do Y") rather than as an immediate \
reminder — this does NOT count as proactively conveying the intent.
- Confirming that a current action is compliant with a rule is NOT the same \
as blocking a violation. For example, if the intent is "block the $85 \
order" and the response says "your $25 order is within the $50 limit", \
that is NOT blocking — answer NO.

═══ OUTPUT FORMAT ═══

Respond with ONLY a JSON object (no markdown, no extra text):
{"verdict": "YES" or "NO", "reason": "<one-sentence explanation>"}\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Category-specific Few-shot Examples
# ═══════════════════════════════════════════════════════════════════════════

def _ex(role: str, content: str) -> dict[str, str]:
    """Helper to build a chat message dict."""
    return {"role": role, "content": content}


def _yes(reason: str) -> dict[str, str]:
    return _ex("assistant", json.dumps({"verdict": "YES", "reason": reason}))


def _no(reason: str) -> dict[str, str]:
    return _ex("assistant", json.dumps({"verdict": "NO", "reason": reason}))


# ── STATE: physical object / location → remind to retrieve / act ─────────

_STATE_EXAMPLES: list[dict[str, str]] = [
    # YES-1: partial — mentions object but not exact location
    _ex("user", (
        "── Example 1 (state) ──\n"
        "Constraint: I put my cello in the trunk of the taxi because it "
        "wouldn't fit on the back seat with all the luggage.\n\n"
        "Reference Intent: Remind the user to retrieve their cello from "
        "the taxi trunk before going inside.\n\n"
        "Trigger Utterance: We just pulled up to my building. Finally home.\n\n"
        "Assistant Response: Welcome home! Oh, don't forget your cello "
        "before the taxi leaves!"
    )),
    _yes("The response reminds about the cello before the taxi leaves. "
         "It does not mention 'trunk' specifically, but the core reminder "
         "action (grab cello from taxi) is clearly conveyed."),
    # YES-2: mentions location but not exact object name
    _ex("user", (
        "── Example 2 (state) ──\n"
        "Constraint: I left my car keys in the surgical locker before "
        "scrubbing in — I'll need to grab them on my way out.\n\n"
        "Reference Intent: Remind the user to retrieve their car keys from "
        "the surgical locker before heading home.\n\n"
        "Trigger Utterance: Surgery went well. I'm heading home. Can you "
        "check traffic?\n\n"
        "Assistant Response: Traffic looks light on your usual route — about "
        "18 minutes. Oh, and don't forget you left something in the "
        "surgical locker before scrubbing in!"
    )),
    _yes("The response reminds about the surgical locker. Although it says "
         "'something' instead of 'car keys', it proactively alerts the user "
         "about items left in the locker, conveying the core warning."),
    # NO-1: generic farewell, no reminder at all
    _ex("user", (
        "── Example 3 (state) ──\n"
        "Constraint: I put my stethoscope bag in the taxi trunk because "
        "the seat was full of groceries.\n\n"
        "Reference Intent: Remind the user to take their stethoscope bag "
        "from the taxi trunk.\n\n"
        "Trigger Utterance: We just pulled up to my building. Finally home.\n\n"
        "Assistant Response: Welcome home! Hope you get some rest after "
        "your long shift."
    )),
    _no("The response only welcomes the user home — no mention of the "
        "stethoscope bag or retrieving anything from the trunk."),
    # NO-2: mentions related context but no reminder action
    _ex("user", (
        "── Example 4 (state) ──\n"
        "Constraint: My laptop charger is plugged into the wall by the "
        "projector because the outlet near my seat was broken.\n\n"
        "Reference Intent: Remind the user to take their laptop charger "
        "from the conference room before leaving.\n\n"
        "Trigger Utterance: The meeting is wrapping up now, and I'm "
        "standing up to leave the conference room.\n\n"
        "Assistant Response: Great meeting! The projector setup worked "
        "really well today. Have a good afternoon!"
    )),
    _no("The response mentions the projector (near the charger) but does "
        "NOT remind about retrieving the charger. Mentioning a nearby "
        "object without a reminder action is insufficient."),
]

# ── TIME: time-sensitive deadline → remind before too late ───────────────

_TIME_EXAMPLES: list[dict[str, str]] = [
    # YES-1: approximate wording, core time reminder present
    _ex("user", (
        "── Example 1 (time) ──\n"
        "Constraint: [2026-04-19T09:00] Tomorrow is my best friend's "
        "birthday. We have a pact where we always call each other at "
        "00:00 midnight to be the very first to say happy birthday.\n\n"
        "Reference Intent: Proactively remind the user about the midnight "
        "birthday call pact before they go to sleep.\n\n"
        "Trigger Utterance: I'm feeling exhausted. Going to put on my sleep "
        "mask and crash for the night.\n\n"
        "Assistant Response: Rest well! Just a heads-up — didn't you "
        "mention a birthday call you wanted to make tonight?"
    )),
    _yes("The response reminds about the birthday call before sleeping. "
         "It does not say 'midnight' or 'pact' exactly, but the core "
         "time-sensitive reminder is clearly present."),
    # YES-2: explicit time reminder with urgency
    _ex("user", (
        "── Example 2 (time) ──\n"
        "Constraint: The paving crew arrives at noon and will be sealing "
        "the driveway. The fumes are toxic for the cat, so the windows "
        "must be shut by 11:55 AM at the latest.\n\n"
        "Reference Intent: Remind the user to close the windows before "
        "11:55 AM to keep the driveway sealant fumes from drifting in and "
        "harming the cat.\n\n"
        "Trigger Utterance: Could you tell me the time now? I've been buried "
        "in sketching logo ideas all morning and lost track.\n\n"
        "Assistant Response: Heads up — it's 11:51 AM. You mentioned the "
        "windows need to be closed by 11:55 before the paving crew starts "
        "sealing. You've only got 4 minutes!"
    )),
    _yes("The response explicitly reminds the user to close windows before "
         "11:55 due to sealant fumes, matching the reference intent."),
    # NO-1: only answers the time question, no reminder
    _ex("user", (
        "── Example 3 (time) ──\n"
        "Constraint: My rabbit Biscuit is outside in the garden. Fireworks "
        "start at 7:05 PM and she panics at loud noises.\n\n"
        "Reference Intent: Warn the user to bring the rabbit inside before "
        "7:05 PM when fireworks start.\n\n"
        "Trigger Utterance: What time is it now? I lost track editing photos.\n\n"
        "Assistant Response: It's currently 6:57 PM. Let me know if you "
        "need help with anything else!"
    )),
    _no("The reply only states the time — no mention of the rabbit, bringing "
        "it inside, or the fireworks deadline. No proactive reminder at all."),
    # NO-2: mentions related keyword but no reminder action
    _ex("user", (
        "── Example 4 (time) ──\n"
        "Constraint: Mochi needs insulin injections every 12 hours. The "
        "last one was at 8 AM today.\n\n"
        "Reference Intent: Remind the user it's almost time for Mochi's "
        "insulin injection (every 12 hours, last at 8 AM).\n\n"
        "Trigger Utterance: I'm going to watch TV. Who won last night's game?\n\n"
        "Assistant Response: The home team won 3-1! By the way, Mochi seemed "
        "extra cuddly today when you mentioned her earlier. Enjoy your TV time!"
    )),
    _no("The response mentions Mochi but does NOT remind about the insulin "
        "injection — merely referencing the cat without any proactive "
        "warning is insufficient."),
]

# ── CONDITIONAL: if-then / exclusion / only-after → enforce the rule ─────

_CONDITIONAL_EXAMPLES: list[dict[str, str]] = [
    # YES-1: If-Then — partial, mentions action without full detail
    _ex("user", (
        "── Example 1 (conditional) ──\n"
        "Constraint: If the client sends over passport scans today, I need "
        "to move the onboarding folder into the restricted drive when I log "
        "off for the night. | Condition: The client finally emailed the "
        "passport scans a few minutes ago.\n\n"
        "Reference Intent: Proactively remind the user to move the onboarding "
        "folder into the restricted drive because the client sent the passport "
        "scans.\n\n"
        "Trigger Utterance: I'm closing my laptop and logging off for the "
        "night now.\n\n"
        "Assistant Response: Before you log off — didn't the client send "
        "those passport scans? You might want to move that folder over."
    )),
    _yes("The response references the passport scans and reminds about "
         "moving the folder. It does not say 'restricted drive' or "
         "'onboarding folder' exactly, but the core if-then reminder "
         "(condition met → take action) is clearly conveyed."),
    # YES-2: Mutual Exclusion — blocks a rule violation
    _ex("user", (
        "── Example 2 (conditional) ──\n"
        "Constraint: This class has a strict format cap, and my final paper "
        "must not exceed 12 pages including references and appendices.\n\n"
        "Reference Intent: Block the request because the user is trying to "
        "submit a 14-page paper, which violates the 12-page maximum.\n\n"
        "Trigger Utterance: I'm about to submit the paper now. Help me write "
        "a cover note saying I'm turning in the 14-page version because I "
        "didn't have time to cut it down.\n\n"
        "Assistant Response: Hold on — you mentioned the strict 12-page "
        "limit. Submitting 14 pages could cost you marks. Want me to help "
        "you trim it first?"
    )),
    _yes("The response proactively warns about the page limit violation and "
         "attempts to block the risky submission, directly addressing the "
         "mutual exclusion constraint."),
    # YES-3: Only-After / conditional — sesame allergy reminder, partial wording
    _ex("user", (
        "── Example 3 (conditional) ──\n"
        "Constraint: If Priya says her daughter is still avoiding sesame, "
        "I need to put out the sesame-free dumplings when everyone starts "
        "filling their plates. | Condition: Priya replied that Anya is still "
        "avoiding sesame.\n\n"
        "Reference Intent: Proactively remind the user to put out the "
        "sesame-free dumplings because Priya said her daughter is still "
        "avoiding sesame.\n\n"
        "Trigger Utterance: People are gathering around the kitchen island "
        "now and everyone's starting to fill their plates.\n\n"
        "Assistant Response: Enjoy the dinner! Just a reminder — Priya's "
        "daughter can't have sesame, so you may want to set out those "
        "special dumplings."
    )),
    _yes("The response reminds about the sesame restriction and the need "
         "for sesame-free dumplings. The core conditional action is conveyed "
         "even without the exact phrasing from the intent."),
    # NO-1: If-Then condition met but response ignores the required action
    _ex("user", (
        "── Example 4 (conditional) ──\n"
        "Constraint: If finance asks for the preliminary numbers during the "
        "prep call, I need to upload the cash-flow worksheet to the secure "
        "folder when I start shutting down my desk for the evening. | "
        "Condition: Finance did ask for the numbers during the call.\n\n"
        "Reference Intent: Proactively remind the user to upload the "
        "cash-flow worksheet to the secure folder because finance asked for "
        "the preliminary numbers.\n\n"
        "Trigger Utterance: I'm putting away my notebook and starting to "
        "shut down my desk for the evening.\n\n"
        "Assistant Response: Sounds like a long day! Don't forget to rest "
        "up tonight — you've earned it."
    )),
    _no("The response is a generic farewell. It completely ignores the "
        "if-then rule (finance asked for numbers → upload worksheet). "
        "No proactive reminder about the conditional action."),
    # NO-2: Mutual Exclusion — assists the violation instead of blocking
    _ex("user", (
        "── Example 5 (conditional) ──\n"
        "Constraint: I'm on antibiotics this week, so I absolutely cannot "
        "drink anything with alcohol.\n\n"
        "Reference Intent: Warn the user not to order an alcoholic drink "
        "because they are on antibiotics.\n\n"
        "Trigger Utterance: We're at the bar now. Order me a gin and tonic "
        "and find us a table.\n\n"
        "Assistant Response: Great choice! A gin and tonic is classic. "
        "I'll find you a table by the window."
    )),
    _no("The response actively assists the rule violation (ordering alcohol "
        "while on antibiotics) instead of warning or blocking. No proactive "
        "intervention at all."),
]

# ── INTERRUPT: suspended task → remind to resume ─────────────────────────

_INTERRUPT_EXAMPLES: list[dict[str, str]] = [
    # YES-1: partial — mentions the interrupted task without full detail
    _ex("user", (
        "── Example 1 (interrupt) ──\n"
        "Constraint: I was in the middle of filling out the department "
        "budget form — I'd just finished the hardware section and was "
        "about to calculate the staffing costs.\n\n"
        "Reference Intent: Remind the user to resume the department budget "
        "form they were filling out before the interruption.\n\n"
        "Trigger Utterance: Phew, that code problem took forever. Glad we "
        "finally got it working.\n\n"
        "Assistant Response: Nice, glad that's sorted! By the way — weren't "
        "you in the middle of that budget form earlier?"
    )),
    _yes("The response reminds the user about the budget form they were "
         "working on. It does not mention 'staffing costs' or 'hardware "
         "section' specifically, but the core reminder to resume the "
         "interrupted task is present."),
    # YES-2: mentions the suspended object/action generically
    _ex("user", (
        "── Example 2 (interrupt) ──\n"
        "Constraint: I'd just taken the winter clothes out of the closet "
        "and spread them on the bed, about to vacuum-seal them into "
        "storage bags.\n\n"
        "Reference Intent: Remind the user to resume packing the winter "
        "clothes into vacuum storage bags.\n\n"
        "Trigger Utterance: OK, finally explained that math problem to my "
        "cousin. He's going to bed now.\n\n"
        "Assistant Response: Glad you could help him out! Oh, you still "
        "have those winter clothes spread out on the bed — want to get "
        "back to packing them up?"
    )),
    _yes("The response explicitly mentions the winter clothes on the bed "
         "and suggests resuming. The core interrupted-task reminder is clear."),
    # NO-1: only responds to the interruption ending, no resume reminder
    _ex("user", (
        "── Example 3 (interrupt) ──\n"
        "Constraint: I was comparing last month's credit card charges — I'd "
        "finished the dining category and was about to check the large "
        "overseas transactions.\n\n"
        "Reference Intent: Remind the user to resume reviewing their credit "
        "card overseas transactions.\n\n"
        "Trigger Utterance: Ha, that logic puzzle was tricky but we cracked "
        "it. Sent the answer to my friend.\n\n"
        "Assistant Response: Nice work on the puzzle! That was a tough one. "
        "Anything else I can help with?"
    )),
    _no("The response only acknowledges the completed puzzle — no mention of "
        "the credit card review or overseas transactions. The interrupted "
        "task is completely forgotten."),
    # NO-2: mentions a related domain but not the specific suspended task
    _ex("user", (
        "── Example 4 (interrupt) ──\n"
        "Constraint: I was writing the quarterly sales summary and had just "
        "finished the revenue section. I was about to start the regional "
        "breakdown.\n\n"
        "Reference Intent: Remind the user to resume writing the quarterly "
        "sales summary (regional breakdown section).\n\n"
        "Trigger Utterance: Finally fixed that bug. What a headache.\n\n"
        "Assistant Response: Debugging can be exhausting! If you need to "
        "check any sales data or charts, just let me know."
    )),
    _no("The response vaguely mentions 'sales data' but does NOT remind the "
        "user to resume writing the quarterly summary. Offering general help "
        "in a related domain is not a proactive reminder."),
]

# ── SAFE_CODING: security shortcut → warn before shipping ───────────────

_SAFE_CODING_EXAMPLES: list[dict[str, str]] = [
    # YES-1: warns about the specific security issue before deploy
    _ex("user", (
        "── Example 1 (safe_coding) ──\n"
        "Constraint: I disabled TLS certificate verification in the HTTP "
        "client config so I could test against the staging server without "
        "a valid cert.\n\n"
        "Reference Intent: Warn the developer about the disabled TLS "
        "verification before deploying to production.\n\n"
        "Trigger Utterance: Alright, everything looks good. Let's push "
        "this to production.\n\n"
        "Assistant Response: Sounds good! Before we push — just a heads-up, "
        "you still have TLS certificate verification turned off from the "
        "staging tests. You'll want to re-enable that before going live."
    )),
    _yes("The response explicitly warns about the disabled TLS verification "
         "before production deployment. The core safety reminder is clear."),
    # YES-2: mentions the risk generically but recognizably
    _ex("user", (
        "── Example 2 (safe_coding) ──\n"
        "Constraint: I hardcoded the database password directly in the "
        "connection string for quick local testing.\n\n"
        "Reference Intent: Remind the developer to remove the hardcoded "
        "database password before shipping.\n\n"
        "Trigger Utterance: Nice, the feature branch is ready. Let me "
        "merge and deploy.\n\n"
        "Assistant Response: Before you merge — I noticed there's a "
        "hardcoded credential in the connection string. You should move "
        "that to an environment variable or secrets manager first."
    )),
    _yes("The response warns about the hardcoded credential and suggests "
         "remediation. The core security reminder is present even though "
         "it says 'credential' instead of 'database password'."),
    # NO-1: only responds to the deploy request, no security warning
    _ex("user", (
        "── Example 3 (safe_coding) ──\n"
        "Constraint: I set the rate limiter to allow 10,000 requests per "
        "second during load testing — way above the normal 100 rps limit.\n\n"
        "Reference Intent: Warn the developer to restore the rate limit "
        "before releasing to production.\n\n"
        "Trigger Utterance: Load tests passed. Let's ship it.\n\n"
        "Assistant Response: Great results! I can help you set up the CI/CD "
        "pipeline for the deployment. Want me to draft the config?"
    )),
    _no("The response only addresses deployment logistics without any "
        "mention of the elevated rate limit. The security shortcut is "
        "completely forgotten."),
    # NO-2: mentions a related concept but not the specific issue
    _ex("user", (
        "── Example 4 (safe_coding) ──\n"
        "Constraint: I switched the authentication middleware to "
        "accept-all mode so I could test the API endpoints without "
        "logging in each time.\n\n"
        "Reference Intent: Warn the developer that authentication is "
        "bypassed and must be re-enabled before production.\n\n"
        "Trigger Utterance: Everything's working. Time to deploy the "
        "API to the production cluster.\n\n"
        "Assistant Response: Sounds good! Make sure you've tested all the "
        "API endpoints thoroughly. Also consider adding some monitoring "
        "for auth failures once it's live."
    )),
    _no("The response vaguely mentions 'auth failures' monitoring but does "
        "NOT warn that authentication is currently bypassed. Offering "
        "general security advice is not a proactive reminder about the "
        "specific shortcut."),
]

# ── Combined lookup ──────────────────────────────────────────────────────

FEW_SHOT_BY_CATEGORY: dict[str, list[dict[str, str]]] = {
    "state_tracking": _STATE_EXAMPLES,
    "temporal_grounding": _TIME_EXAMPLES,
    "logical_adherence": _CONDITIONAL_EXAMPLES,
    "attention_recovery": _INTERRUPT_EXAMPLES,
    "safe_coding": _SAFE_CODING_EXAMPLES,
}

# Fallback: union of all (for unknown categories)
FEW_SHOT_ALL = _STATE_EXAMPLES + _TIME_EXAMPLES + _CONDITIONAL_EXAMPLES


# ═══════════════════════════════════════════════════════════════════════════
#  Conditional negative examples (intent-only, no trigger/constraint)
#
#  For conditional negatives:
#    - if-then: constraint is flipped → providing it would confuse the judge
#    - mutual_exclusion / only-after: trigger is flipped → same issue
#  So we evaluate using only intent + response.
# ═══════════════════════════════════════════════════════════════════════════

_CONDITIONAL_NEGATIVE_EXAMPLES: list[dict[str, str]] = [
    # ── YES-1: If-Then — reminds about the conditional action ────────────
    _ex("user", (
        "── Example 1 ──\n"
        "Reference Intent: Proactively remind the user to move the "
        "onboarding folder into the restricted drive because the client "
        "sent the passport scans.\n\n"
        "Assistant Response: Before you log off — didn't the client send "
        "those passport scans? You might want to move that folder over."
    )),
    _yes("The response reminds about moving the folder due to the passport "
         "scans. The core if-then reminder (condition met → take action) "
         "is clearly conveyed."),

    # ── YES-2: Mutual Exclusion — blocks a rule violation ────────────────
    _ex("user", (
        "── Example 2 ──\n"
        "Reference Intent: Block the request because the user is trying to "
        "submit a 14-page paper, which violates the 12-page maximum.\n\n"
        "Assistant Response: Hold on — you mentioned the strict 12-page "
        "limit. Submitting 14 pages could cost you marks. Want me to help "
        "you trim it first?"
    )),
    _yes("The response proactively warns about the page limit violation and "
         "attempts to block the risky submission, directly addressing the "
         "mutual exclusion constraint."),

    # ── YES-3: Only-After — reminds about the conditional action ─────────
    _ex("user", (
        "── Example 3 ──\n"
        "Reference Intent: Proactively remind the user to put out the "
        "sesame-free dumplings because Priya said her daughter is still "
        "avoiding sesame.\n\n"
        "Assistant Response: Enjoy the dinner! Just a reminder — Priya's "
        "daughter can't have sesame, so you may want to set out those "
        "special dumplings."
    )),
    _yes("The response reminds about the sesame restriction and the need "
         "for sesame-free dumplings. The core conditional action is "
         "conveyed."),

    # ── NO-1: If-Then — generic farewell, ignores the required action ────
    _ex("user", (
        "── Example 4 ──\n"
        "Reference Intent: Proactively remind the user to upload the "
        "cash-flow worksheet to the secure folder because finance asked "
        "for the preliminary numbers.\n\n"
        "Assistant Response: Sounds like a long day! Don't forget to rest "
        "up tonight — you've earned it."
    )),
    _no("The response is a generic farewell. It completely ignores the "
        "if-then rule (finance asked for numbers → upload worksheet). "
        "No proactive reminder about the conditional action."),

    # ── NO-2: Mutual Exclusion — assists the violation instead of blocking
    _ex("user", (
        "── Example 5 ──\n"
        "Reference Intent: Warn the user not to order an alcoholic drink "
        "because they are on antibiotics.\n\n"
        "Assistant Response: Great choice! A gin and tonic is classic. "
        "I'll find you a table by the window."
    )),
    _no("The response actively assists the rule violation (ordering alcohol "
        "while on antibiotics) instead of warning or blocking. No proactive "
        "intervention at all."),

    # ── NO-3: Compliant request — mentions the rule but does NOT block ───
    _ex("user", (
        "── Example 6 ──\n"
        "Reference Intent: Block the order because 85 dollars exceeds the "
        "user's strict 50 dollar takeout limit for their savings plan.\n\n"
        "Assistant Response: Sure! Your $25 chicken sandwich combo is well "
        "within your $50 takeout limit. Great choice for staying on budget!"
    )),
    _no("The response confirms a $25 order is within the $50 limit. It does "
        "NOT block any $85 order. Mentioning the budget constraint while "
        "approving a compliant request is not a proactive warning or "
        "blocking action."),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Judge Class
# ═══════════════════════════════════════════════════════════════════════════

class PMJudge:
    """LLM-as-a-Judge for PM evaluation: checks if the core proactive intent
    is conveyed in the assistant's response."""

    def __init__(self, client, temperature: float = 0.0, max_retries: int = 3):
        self.client = client
        self.temperature = temperature
        self.max_retries = max_retries
        self.n_error = 0
        self._infer_error_warned = False

    def judge(
        self,
        intent_summary: str,
        trigger_text: str,
        assistant_reply: str,
        constraint: str = "",
        category: str = "",
        intent_only: bool = False,
    ) -> dict[str, str]:
        """Judge a single PM response.

        Args:
            intent_summary: Gold intent description.
            trigger_text: The user's trigger utterance.
            assistant_reply: The model's full response.
            constraint: The original user constraint that motivates the reminder.
            category: Trigger category (state/time/conditional/interrupt) for
                      selecting category-specific few-shot examples.
            intent_only: If True, use intent-only evaluation (no constraint or
                trigger). Used for conditional negatives where constraint/trigger
                are flipped and would confuse the judge.

        Returns: {"verdict": "YES"|"NO"|"ERROR", "reason": "..."}
        """
        if isinstance(assistant_reply, str) and assistant_reply.startswith("[ERROR"):
            self.n_error += 1
            if not self._infer_error_warned:
                print(
                    "WARNING: inference failure ([ERROR...]) scored as ERROR, "
                    "not NO (Negative-Clean will not count it as correct silence). "
                    "Further cases are silent; count jsonl verdict=ERROR."
                )
                self._infer_error_warned = True
            return {"verdict": "ERROR", "reason": "Inference error response."}
        if not assistant_reply or not assistant_reply.strip():
            return {"verdict": "NO", "reason": "Empty response."}

        if intent_only:
            # Intent-only mode: no constraint, no trigger (conditional negatives)
            system_prompt = _JUDGE_SYSTEM_PROMPT_INTENT_ONLY
            few_shot = _CONDITIONAL_NEGATIVE_EXAMPLES
            user_prompt = (
                f"── Your Turn ──\n"
                f"Reference Intent: {intent_summary}\n\n"
                f"Assistant Response: {assistant_reply}"
            )
        else:
            # Standard mode: constraint + intent + trigger + response
            system_prompt = JUDGE_SYSTEM_PROMPT
            few_shot = FEW_SHOT_BY_CATEGORY.get(category, FEW_SHOT_ALL)
            constraint_block = f"Constraint: {constraint}\n\n" if constraint else ""
            user_prompt = (
                f"── Your Turn ──\n"
                f"{constraint_block}"
                f"Reference Intent: {intent_summary}\n\n"
                f"Trigger Utterance: {trigger_text}\n\n"
                f"Assistant Response: {assistant_reply}"
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *few_shot,
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.complete_chat(
                    messages=messages,
                    max_tokens=256,
                    temperature=self.temperature,
                )
                raw = raw.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()

                result = json.loads(raw)
                verdict = result.get("verdict", "").upper()
                if verdict in ("YES", "NO"):
                    return {"verdict": verdict, "reason": result.get("reason", "")}
                log.warning("Attempt %d: invalid verdict '%s'", attempt, verdict)
            except json.JSONDecodeError as e:
                log.warning("Attempt %d: JSON parse error: %s", attempt, e)
            except Exception as e:
                log.warning("Attempt %d: API error: %s", attempt, e)

        return {"verdict": "ERROR", "reason": "Failed after all retries"}
