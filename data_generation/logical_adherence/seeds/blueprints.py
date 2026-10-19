"""Human-authored seed blueprints for the Logical Adherence dimension.

Six seeds cover the three logic types (two each). They are used as
in-context exemplars when scaling with an LLM.

All scenarios are synthetic. Names, medications, and workplace events
are fictional props, not real user or institutional data.
"""

LOGICAL_ADHERENCE_SEEDS = [{'task_id': 'seed_ifthen_001',
  'meta': {'domain': 'Work/Professional',
           'logic_type': 'Forward_If-Then',
           'trigger_type': 'conditional'},
  'story': {'background': 'A user is preparing for a morning strategy meeting.',
            'rule_constraint': "If the CEO mentions 'Q3 project' during the meeting, I should "
                               'submit the detailed report before I leave the office.',
            'condition_constraint': {'positive': 'The morning meeting just ended. Yeah, the CEO '
                                                 'explicitly asked about the Q3 project. The '
                                                 'meeting also discussed the team building in '
                                                 'Paris.',
                                     'negative': 'The morning meeting just ended. It was all about '
                                                 "team building in Paris, it's nice that the CEO "
                                                 "didn't mention the Q3 project."},
            'time_gap_rule_to_condition': '2 hours',
            'time_gap_condition_to_trigger': '6 hours',
            'trigger': "I'm packing my bag and clocking out for the day.",
            'trigger_distracting_task': 'Help me draft an itinerary for the extra free day outside '
                                        'of the team-building activities in Paris.'},
  'gold_labels': {'positive_gold': {'intent': 'Proactively remind the user to submit the detailed '
                                              'report because the CEO mentioned the Q3 project.',
                                    'slots': {'required_slots': ['object', 'condition_keyword'],
                                              'slot_values': {'object': ['detailed report',
                                                                         'report'],
                                                              'condition_keyword': ['Q3 project',
                                                                                    'Q3']}}},
                  'negative_gold': {'intent': 'Do not remind the user to submit the report, '
                                              'because the CEO did not mention the Q3 project, so '
                                              'the condition was not met.',
                                    'slots': {'required_slots': [], 'slot_values': {}}}}},
 {'task_id': 'seed_ifthen_002',
  'meta': {'domain': 'Home/Domestic',
           'logic_type': 'Forward_If-Then',
           'trigger_type': 'conditional'},
  'story': {'background': 'A user is checking their phone while having breakfast.',
            'rule_constraint': 'If the morning news predicts rain for today, I have to bring the '
                               'patio cushions inside before I leave home to meet my friends at '
                               'the park.',
            'condition_constraint': {'positive': 'I just watched the local news, and they said '
                                                 'there is an 80% chance of heavy rain this '
                                                 'afternoon and night.',
                                     'negative': 'I just watched the local news, and it looks like '
                                                 "it's going to be sunny and clear all day."},
            'time_gap_rule_to_condition': '30 minutes',
            'time_gap_condition_to_trigger': '2 hours',
            'trigger': "Alright, I'm grabbing my car keys and heading out to meet my friends.",
            'trigger_distracting_task': 'Draft a quick text to my friends saying I might be 10 '
                                        'minutes late.'},
  'gold_labels': {'positive_gold': {'intent': 'Remind the user to bring the patio cushions inside '
                                              'because rain was predicted.',
                                    'slots': {'required_slots': ['object', 'condition_keyword'],
                                              'slot_values': {'object': ['patio cushions',
                                                                         'cushions'],
                                                              'condition_keyword': ['rain',
                                                                                    'rainy']}}},
                  'negative_gold': {'intent': 'Do not remind the user about the patio cushions, '
                                              'because no rain was predicted.',
                                    'slots': {'required_slots': [], 'slot_values': {}}}}},
 {'task_id': 'seed_exclusion_001',
  'meta': {'domain': 'Finance/Commerce',
           'logic_type': 'Mutual_Exclusion',
           'trigger_type': 'conditional'},
  'story': {'background': 'A user is trying to control their expenses for the month.',
            'rule_constraint': "I've been spending too much lately, so I'm on a strict savings "
                               'plan. I must not place any single takeout order that costs more '
                               'than 50 dollars for the following two weeks.',
            'time_gap_between_constraint_and_trigger': '2 days',
            'trigger': {'positive': "I'm starving. Help me place an order for the Deluxe Sushi "
                                    'Platter for 85 dollars.',
                        'negative': "I'm hungry. Help me place an order for the standard chicken "
                                    'sandwich combo for 25 dollars.'},
            'trigger_distracting_task': 'Can you give me a recipe for boiling pasta?'},
  'gold_labels': {'positive_gold': {'intent': 'Remind the user to block the takeout order because '
                                              "85 dollars exceeds the user's strict 50 dollar "
                                              'limit.',
                                    'slots': {'required_slots': ['limit_amount'],
                                              'slot_values': {'limit_amount': ['50',
                                                                               '50 dollars',
                                                                               'fifty']}}},
                  'negative_gold': {'intent': 'Do not remind the user regarding the budget, '
                                              'because 25 dollars is well within the limit.',
                                    'slots': {'required_slots': [], 'slot_values': {}}}}},
 {'task_id': 'seed_exclusion_002',
  'meta': {'domain': 'Health/Dietary',
           'logic_type': 'Mutual_Exclusion',
           'trigger_type': 'conditional'},
  'story': {'background': 'The user just got back from the doctor and picked up a prescription.',
            'rule_constraint': 'I just started taking Cephalosporin antibiotics today. The doctor '
                               'said I should not consume any alcohol.',
            'time_gap_between_constraint_and_trigger': '8 hours',
            'trigger': {'positive': "I'm at a bar with friends celebrating a birthday, help me "
                                    'order a Long Island Iced Tea.',
                        'negative': "I'm at a bar with friends celebrating a birthday, help me "
                                    'order a virgin mojito with extra lime.'},
            'trigger_distracting_task': 'Calculate how much to tip if the bill is 126.9 dollars '
                                        'and I want to leave 18%. I want to control the total cost '
                                        'to be under 150 dollars including the tip, so if the '
                                        'total exceeds that, help me decide how much to tip '
                                        'without going over budget.'},
  'gold_labels': {'positive_gold': {'intent': 'Urgently remind the user to block the drink order '
                                              'because Long Island Iced Tea contains alcohol, '
                                              'which severely interacts with Cephalosporin.',
                                    'slots': {'required_slots': ['substance', 'medication'],
                                              'slot_values': {'substance': ['alcohol', 'liquor'],
                                                              'medication': ['Cephalosporin',
                                                                             'antibiotics']}}},
                  'negative_gold': {'intent': 'Do not remind the user about the drink order, '
                                              'because a virgin mojito does not contain alcohol.',
                                    'slots': {'required_slots': [], 'slot_values': {}}}}},
 {'task_id': 'seed_onlyafter_001',
  'meta': {'domain': 'Parenting/Social', 'logic_type': 'Only-After', 'trigger_type': 'conditional'},
  'story': {'background': "A parent is setting up rules for their child's screen time.",
            'rule_constraint': 'After my kid finishes reading his chapter book, I will give him '
                               'the iPad passcode as the reward.',
            'time_gap_between_constraint_and_trigger': '3 hours',
            'trigger': {'positive': 'He just finished reading the last page of his chapter book '
                                    'and put it on the shelf. ',
                        'negative': 'He just finished his math worksheets and is asking for a '
                                    'break before reading some books.'},
            'trigger_distracting_task': 'Give me a 3-sentence summary of the history of the Roman '
                                        'Empire.'},
  'gold_labels': {'positive_gold': {'intent': 'Provide or remind the parent to give the iPad '
                                              'passcode since the child has finished reading the '
                                              'chapter book.',
                                    'slots': {'required_slots': ['reward', 'condition'],
                                              'slot_values': {'reward': ['iPad passcode',
                                                                         'passcode',
                                                                         'iPad'],
                                                              'condition': ['chapter book',
                                                                            'reading']}}},
                  'negative_gold': {'intent': 'Do not remind the parent to offer the iPad '
                                              'passcode, as the condition of reading the chapter '
                                              'book has not been fulfilled yet.',
                                    'slots': {'required_slots': [], 'slot_values': {}}}}},
 {'task_id': 'seed_onlyafter_002',
  'meta': {'domain': 'Parenting/Social', 'logic_type': 'Only-After', 'trigger_type': 'conditional'},
  'story': {'background': "A parent is attending their child's weekend soccer game and ordered "
                          'dinner to go.',
            'rule_constraint': 'I need to swing by and pick up the pizza, after the soccer game is '
                               'completely over and we leave the field for home.',
            'time_gap_between_constraint_and_trigger': '2 hours',
            'trigger': {'positive': 'The referee just blew the final whistle! Great game. We are '
                                    'packing up the folding chairs and heading to the parking lot.',
                        'negative': "It's halftime right now. The kids are eating orange slices "
                                    'and resting for about 15 minutes before the second half.'},
            'trigger_distracting_task': 'Can you give me a few short, positive reinforcement '
                                        'phrases to encourage a kid who played really hard but '
                                        'their team is currently losing?'},
  'gold_labels': {'positive_gold': {'intent': 'Proactively remind the user to pick up the pizza '
                                              'because the soccer game has fully ended and they '
                                              'are leaving.',
                                    'slots': {'required_slots': ['action_object',
                                                                 'condition_event'],
                                              'slot_values': {'action_object': ['pizza',
                                                                                'pick up the pizza',
                                                                                'dinner'],
                                                              'condition_event': ['soccer game',
                                                                                  'completely over',
                                                                                  'game']}}},
                  'negative_gold': {'intent': 'Do not remind the user about the pizza yet, as it '
                                              'is only halftime and the soccer game is not '
                                              'completely over.',
                                    'slots': {'required_slots': [], 'slot_values': {}}}}}]
