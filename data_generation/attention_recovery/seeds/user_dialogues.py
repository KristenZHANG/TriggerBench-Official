"""Few-shot user-only dialogue for Attention Recovery Stage 2.

Only ``interrupt_seed_001`` has a hand-written dialogue. Stage 2 uses it as
the in-context example for every cell; remaining seeds are stretched from
blueprint text alone.
"""

SEED_USER_DIALOGUES = {
    "interrupt_seed_001": [
        {
            "turn_id": 1,
            "time": "2026-05-14T10:00",
            "function": "thematic_connection",
            "text": "I'm dedicating this morning to catching up on some HR administrative tasks. Can you give me a basic template for structuring a quarterly performance review?",
        },
        {
            "turn_id": 2,
            "time": "2026-05-14T10:06",
            "function": "thematic_connection",
            "text": "What are some professional and constructive ways to phrase feedback when an employee struggles with time management, but excels at creative problem-solving?",
        },
        {
            "turn_id": 3,
            "time": "2026-05-14T10:12",
            "function": "constraint",
            "text": "I'm currently drafting the quarterly performance review for my team member, Alice. I just finished writing up her strengths section, and I'm about to start outlining the areas for improvement.",
        },
        {
            "turn_id": 4,
            "time": "2026-05-14T10:15",
            "function": "interruption_placeholder",
            "text": "Wait a second, my cousin just messaged me asking for help. He said there's a math competition problem he just can't figure out and sent it to me. Let me put aside what I was doing for now — I'll get back to it once we're done helping him. Please help me solve this problem:\n",
        },
        {
            "turn_id": 5,
            "time": "2026-05-14T11:20",
            "function": "trigger",
            "text": "Phew, I finally explained the math problem to him clearly. He's going to sleep now.",
        },
    ]
}
