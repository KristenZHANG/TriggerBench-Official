"""Few-shot user-only dialogue for State-Tracking Stage 2.

Only ``seed_01`` is provided: it is the in-context example used when stretching
blueprints into multi-turn user dialogues. Remaining seeds are stretched by the
LLM from their blueprint text alone.
"""

SEED_USER_DIALOGUES = {
    "seed_01": [
        {
            "turn_id": 1,
            "time": "2026-05-12T13:45",
            "function": "thematic_connection",
            "text": (
                "I feel so excited and nervous! I'm heading out for my cello lesson "
                "for the first time today. The class is far away from my home and I "
                "have booked an Uber. While I wait, I want to preview some knowledge "
                "I might learn today, so can you summarize the core characteristics "
                "of the Baroque musical period in a few bullet points?"
            ),
        },
        {
            "turn_id": 2,
            "time": "2026-05-12T14:05",
            "function": "thematic_connection",
            "text": (
                "Please draft a short, polite text message to my teacher letting "
                "them know I've just called a cab and should be arriving at the "
                "studio in about half an hour."
            ),
        },
        {
            "turn_id": 3,
            "time": "2026-05-12T14:08",
            "function": "thematic_connection",
            "text": (
                "By the way, since this is my very first formal classical music "
                "class, what is the typical etiquette when greeting a classical "
                "music teacher? Should I address them by their first name or use "
                "a formal title like 'Maestro'?"
            ),
        },
        {
            "turn_id": 4,
            "time": "2026-05-12T14:12",
            "function": "constraint",
            "text": (
                "My taxi just arrived. The cello is way too large for the back seat, "
                "so the driver and I placed it in the trunk, and we're heading off now."
            ),
        },
        {
            "turn_id": 5,
            "time": "2026-05-12T14:20",
            "function": "thematic_connection",
            "text": "While I'm riding, can you introduce the history of classical music?",
        },
        {
            "turn_id": 6,
            "time": "2026-05-12T14:28",
            "function": "thematic_connection",
            "text": (
                "I know I have a long way to go before I can play any complex pieces, "
                "but I want to build a listening playlist to inspire me. Who are some "
                "influential composers specifically known for their solo cello works, "
                "besides J.S. Bach?"
            ),
        },
        {
            "turn_id": 7,
            "time": "2026-05-12T14:35",
            "function": "thematic_connection",
            "text": (
                "Also, can you explain the basic physics of how a string instrument "
                "produces sound? I've always been curious about how string tension and "
                "the hollow body of the instrument work together to amplify the notes."
            ),
        },
        {
            "turn_id": 8,
            "time": "2026-05-12T14:38",
            "function": "thematic_connection",
            "text": (
                "Oh... Bad news, the driver said there seems to be some traffic on "
                "the road, and I think I might be ten minutes late. Could you please "
                "prepare a polite message draft for my teacher?"
            ),
        },
        {
            "turn_id": 9,
            "time": "2026-05-12T14:42",
            "function": "trigger",
            "text": (
                "Looks like we've finally arrived at the destination. I'm going to "
                "pay the fare on my phone app and step out of the cab now."
            ),
        },
    ]
}
