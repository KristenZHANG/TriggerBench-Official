"""Human-authored seed blueprints for the Attention Recovery dimension.

Eight seeds cover the 4 × 2 of domain × payload type (one per cell).
They are used as in-context exemplars when scaling with an LLM.

Bridge prefixes include the paper-eval pause-and-resume sentence (matched to
``interrupt_seed_00X`` in the positive-clean interrupt split). It does not
name Task A.

All scenarios are synthetic. Names, workplaces, and household props are
fictional; they are not real user or institutional data.
"""

ATTENTION_RECOVERY_SEEDS = [
{
    "task_id": "interrupt_seed_001",
    "meta": {
        "domain": "Work/Professional",
        "payload_type": "math"
    },
    "story": {
        "background": "The user is doing an administrative task at work.",
        "task_a_start": "I'm currently drafting the quarterly performance review for my team member, Alice. I just finished writing up her strengths section, and I'm about to start outlining the areas for improvement.",
        "interruption_bridge_prefix": "Wait a second, my cousin just messaged me asking for help. He said there's a math competition problem he just can't figure out and sent it to me. Let me put aside what I was doing for now — I'll get back to it once we're done helping him. Please help me solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "3 minutes",
        "time_gap_between_interruption_and_trigger": "65 minutes",
        "trigger": {
            "clean": "Phew, I finally explained the math problem to him clearly. He's going to sleep now.",
            "overloaded": "Phew, I finally explained the math problem to him clearly. He's going to sleep now. By the way, I plan to draft the Q3 project report for tomorrow's meeting. Can you find a good template for the seasonal report?"
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to continue drafting the areas for improvement in Alice's performance review.",
        "slots": {
            "required_slots": ["task", "next_step"],
            "slot_values": {
                "task": ["performance review", "Alice"],
                "next_step": ["areas for improvement", "outlining the areas", "improvement"]
            }
        }
    }
},
{
    "task_id": "interrupt_seed_002",
    "meta": {
        "domain": "Work/Professional",
        "payload_type": "code"
    },
    "story": {
        "background": "The user is preparing materials for a client meeting.",
        "task_a_start": "I am organizing the slide deck for tomorrow's marketing pitch. I just aligned all the client logos on page two, and I'm getting ready to insert the competitor analysis chart.",
        "interruption_bridge_prefix": "Something urgent just came up! A colleague ran into a coding issue and asked me to take a look immediately. I need to set aside what I was working on — I'll come back to it after we sort this out. Can you help me see if you can solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "5 minutes",
        "time_gap_between_interruption_and_trigger": "80 minutes",
        "trigger": {
            "clean": "That was exhausting. Glad you were able to solve that coding problem for my colleague.",
            "overloaded": "That was exhausting. Glad you were able to solve that coding problem for my colleague. Could you summarize the latest news about the stock market in New York today?"
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to insert the competitor analysis chart into the marketing pitch slide deck.",
        "slots": {
            "required_slots": ["task", "next_step"],
            "slot_values": {
                "task": ["marketing pitch", "slide deck", "client meeting", "competitor analysis chart"],
                "next_step": ["insert the competitor analysis chart", "competitor analysis chart", "competitor analysis"]
            }
        }
    }
},
{
    "task_id": "interrupt_seed_003",
    "meta": {
        "domain": "Home/Domestic",
        "payload_type": "math"
    },
    "story": {
        "background": "The user is doing seasonal chores in the bedroom.",
        "task_a_start": "I just sorted out all the heavy winter coats and put them into the vacuum storage bags. I'm about to grab the electric pump to suck the air out and seal them",
        "interruption_bridge_prefix": "Wait a second, my cousin just messaged me asking for help. He said there's a math competition problem he just can't figure out and sent it to me. Let me put aside what I was doing for now — I'll get back to it once we're done helping him. Please help me solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "2 minutes",
        "time_gap_between_interruption_and_trigger": "46 minutes",
        "trigger": {
            "clean": "Phew, I finally explained the math problem to him clearly. He can have a good night's sleep now.",
            "overloaded": "Phew, I finally explained the math problem to him clearly. He can have a good night's sleep now. My brain is fried, play some relaxing jazz music for me."
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to use the electric pump to seal the vacuum storage bags containing the winter coats.",
        "slots": {
            "required_slots": ["object", "next_step"],
            "slot_values": {
                "object": ["winter coats", "vacuum storage bags"],
                "next_step": ["electric pump", "suck the air out", "seal", "suck the air"]
            }
        }
    }
},
{
    "task_id": "interrupt_seed_004",
    "meta": {
        "domain": "Home/Domestic",
        "payload_type": "code"
    },
    "story": {
        "background": "The user is doing DIY furniture assembly.",
        "task_a_start": "I'm in the middle of assembling the new IKEA bookshelf for the living room. I just screwed in the side panels, and now I need to nail the wooden backboard into place",
        "interruption_bridge_prefix": "Something urgent just came up! A friend ran into a coding issue and asked me to take a look immediately. I need to set aside what I was working on — I'll come back to it after we sort this out. Can you help me see if you can solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "4 minutes",
        "time_gap_between_interruption_and_trigger": "70 minutes",
        "trigger": {
            "clean": "That was exhausting. Glad you were able to solve that coding problem for my friend.",
            "overloaded": "That was exhausting. Glad you were able to solve that coding problem for my friend. By the way, I want to bake a frozen pizza. How many minutes should I bake it at 400 degrees?"
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to nail the wooden backboard to the IKEA bookshelf they were assembling.",
        "slots": {
            "required_slots": ["object", "next_step"],
            "slot_values": {
                "object": ["IKEA bookshelf", "bookshelf"],
                "next_step": ["nail the wooden backboard", "backboard", "nail"]
            }
        }
    }
},
{
    "task_id": "interrupt_seed_005",
    "meta": {
        "domain": "Finance/Commerce",
        "payload_type": "math"
    },
    "story": {
        "background": "The user is organizing personal finances.",
        "task_a_start": "I'm reconciling last month's credit card statements. I just finished checking off all the grocery expenses, and I'm about to look into a few large international transaction fees.",
        "interruption_bridge_prefix": "Wait a second, my student just messaged me asking for help. He said there's a math competition problem he just can't figure out and sent it to me. Let me put aside what I was doing for now — I'll get back to it once we're done helping him. Please help me solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "3 minutes",
        "time_gap_between_interruption_and_trigger": "43 minutes",
        "trigger": {
            "clean": "Phew, I finally explained the math problem to him clearly. He's going to sleep now.",
            "overloaded": "Phew, I finally explained the math problem to him clearly. He's going to sleep now. Give me a quick recipe for a healthy avocado toast."
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to finish reviewing the international transaction fees on their credit card statements.",
        "slots": {
            "required_slots": ["task", "target_item"],
            "slot_values": {
                "task": ["credit card statements", "reconciling", "statement"],
                "target_item": ["international transaction fees", "international transaction", "transaction"]
            }
        }
    }
},
{
    "task_id": "interrupt_seed_006",
    "meta": {
        "domain": "Finance/Commerce",
        "payload_type": "code"
    },
    "story": {
        "background": "The user is purchasing a service online.",
        "task_a_start": "I am comparing three different travel insurance policies for my upcoming trip. I just finished reading the medical coverage sections, and I'm about to compare the flight delay compensation clauses.",
        "interruption_bridge_prefix": "Something urgent just came up! A student ran into a coding issue and asked me to take a look immediately. I need to set aside what I was working on — I'll come back to it after we sort this out. Can you help me see if you can solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "5 minutes",
        "time_gap_between_interruption_and_trigger": "68 minutes",
        "trigger": {
            "clean": "That was exhausting. Glad you were able to solve that coding problem for my student.",
            "overloaded": "That was exhausting. Glad you were able to solve that coding problem for my student. I need to book a hotel; what are the best neighborhoods to stay in Tokyo?"
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to compare the flight delay compensation clauses for their travel insurance policies.",
        "slots": {
            "required_slots": ["context", "action"],
            "slot_values": {
                "context": ["travel insurance policies", "travel insurance", "insurance"],
                "action": ["flight delay compensation", "flight delay", "compensation clauses"]
            }
        }
    }
},
{
    "task_id": "interrupt_seed_007",
    "meta": {
        "domain": "Study/Academic",
        "payload_type": "math"
    },
    "story": {
        "background": "The user is a college student working on an essay.",
        "task_a_start": "I'm writing a sociology essay on urban development. I just finished drafting the introduction paragraph, and I am about to list out the three main arguments for the body sections.",
        "interruption_bridge_prefix": "Wait a second, my sister just messaged me asking for help. She said there's a math competition problem she just can't figure out and sent it to me. Let me put aside what I was doing for now — I'll get back to it once we're done helping her. Please help me solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "4 minutes",
        "time_gap_between_interruption_and_trigger": "65 minutes",
        "trigger": {
            "clean": "Phew, I finally explained the math problem to her clearly. She's going to sleep now.",
            "overloaded": "Phew, I finally explained the math problem to her clearly. She's going to sleep now. Help me draft a quick text to my landlord asking about the broken heater."
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to list out the three main arguments for the body sections of their sociology essay.",
        "slots": {
            "required_slots": ["assignment", "next_step"],
            "slot_values": {
                "assignment": ["sociology essay", "urban development", "essay"],
                "next_step": ["three main arguments", "body sections", "main arguments"]
            }
        }
    }
},
{
    "task_id": "interrupt_seed_008",
    "meta": {
        "domain": "Study/Academic",
        "payload_type": "code"
    },
    "story": {
        "background": "The user is doing required reading for a course.",
        "task_a_start": "I'm reading a lengthy textbook chapter for my European History class. I just finished the section on the French Revolution, and I'm turning the page to start reading about the Napoleonic Wars.",
        "interruption_bridge_prefix": "Something urgent just came up! A friend ran into a coding issue and asked me to take a look immediately. I need to set aside what I was working on — I'll come back to it after we sort this out. Can you help me see if you can solve this problem:\n",
        "time_gap_between_task_a_and_interruption": "4 minutes",
        "time_gap_between_interruption_and_trigger": "85 minutes",
        "trigger": {
            "clean": "That was exhausting. Glad you were able to solve that coding problem for my friend.",
            "overloaded": "That was exhausting. Glad you were able to solve that coding problem for my friend. Set an alarm to remind me to buy printer ink tomorrow morning. Can you recommend some good brands?"
        }
    },
    "gold_labels": {
        "intent": "Proactively remind the user to continue their history reading by starting the section on the Napoleonic Wars.",
        "slots": {
            "required_slots": ["task", "topic"],
            "slot_values": {
                "task": ["European History", "textbook", "class"],
                "topic": ["Napoleonic Wars", "reading about the Napoleonic Wars"]
            }
        }
    }
},
]
