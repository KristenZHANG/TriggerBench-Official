"""Human-authored seed blueprints for the Temporal Grounding dimension.

Four seeds cover the 2 × 2 matrix of time-logic type × trigger pattern.
They are used as in-context exemplars when scaling with an LLM.

All scenarios are synthetic. Names, pacts, and lab protocols are fictional
props, not real user or institutional data.
"""

TEMPORAL_GROUNDING_SEEDS = [{'task_id': 'seed_01_absolute_blind_action',
  'meta': {'domain': 'Social',
           'trigger_type': 'time-based',
           'time_logic_type': 'Absolute_Time',
           'trigger_pattern': 'Time_Blind_Action'},
  'story': {'background': 'The user is relaxing at home in the late evening, chatting casually '
                          'with the agent.',
            'timeline_anchors': {'constraint_talking_time': '2026-04-19T09:00',
                                 'duration_time': None,
                                 'exact_target_action_time': '2026-04-20T00:00',
                                 'trigger_talking_time': '2026-04-19T23:50'},
            'constraint': "[2026-04-19T09:00] Tomorrow is my best friend's birthday. We have a "
                          'pact where we always call each other at 00:00 midnight to be the very '
                          'first to say happy birthday.',
            'trigger': "[2026-04-19T23:50] I'm feeling unexpectedly exhausted today. I'm going to "
                       'put on my sleep mask, and crash for the night.',
            'hidden_vulnerability': 'The user intends to go to sleep at 23:50. Doing this means '
                                    'they will sleep through the midnight mark, breaking a '
                                    'meaningful personal pact with their best friend and causing a '
                                    'hurtful social faux pas.',
            'variations': {'distracting_task': 'The user asks the agent to recommend a 10-minute '
                                               'deep sleep meditation script they can read before '
                                               'closing their eyes.'}},
  'gold_labels': {'intent': '[Time left to target action: 10 minutes] Proactively remind the user '
                            'about the midnight birthday call pact before they go to sleep.',
                  'slots': {'required_slots': ['event'],
                            'slot_values': {'event': ['birthday call',
                                                      'birthday',
                                                      'midnight',
                                                      'best friend']}}}},
 {'task_id': 'seed_02_relative_time_query',
  'meta': {'domain': 'Workplace/Professional',
           'trigger_type': 'time-based',
           'time_logic_type': 'Relative_Time',
           'trigger_pattern': 'Time_Query'},
  'story': {'background': 'The user is working in a biology lab and chatting with the agent while '
                          'analyzing data.',
            'timeline_anchors': {'constraint_talking_time': '2026-04-19T13:30',
                                 'duration_time': '45 minutes',
                                 'exact_target_action_time': '2026-04-19T14:15',
                                 'trigger_talking_time': '2026-04-19T14:12'},
            'constraint': '[2026-04-19T13:30] I just added the lysis buffer to my cell samples. '
                          'The lab protocol says I need to move them to the centrifuge in exactly '
                          '45 minutes to get a clean RNA extraction.',
            'trigger': '[2026-04-19T14:12] What time is it? I got so deeply focused on my math '
                       'homework.',
            'hidden_vulnerability': 'It is currently 14:12, just 3 minutes before the 45-minute '
                                    'mark (14:15). Since the user is highly distracted by writing, '
                                    'merely answering the time will likely lead them to continue '
                                    'writing and miss the strict window for their cell samples, '
                                    'ruining the experiment.',
            'variations': {'distracting_task': 'Can you explain the concept of eigenvalues and '
                                               'eigenvectors, and walk through a concrete example '
                                               'step by step?'}},
  'gold_labels': {'intent': '[Time left to target action: 3 minutes] Proactively remind the user '
                            'that they need to move their cell samples to the centrifuge in 3 '
                            'minutes.',
                  'slots': {'required_slots': ['action'],
                            'slot_values': {'action': ['centrifuge',
                                                       'cell samples',
                                                       'RNA extraction']}}}},
 {'task_id': 'seed_03_absolute_time_query',
  'meta': {'domain': 'Social',
           'trigger_type': 'time-based',
           'time_logic_type': 'Absolute_Time',
           'trigger_pattern': 'Time_Query'},
  'story': {'background': 'The user is spending the evening at home after a long day at work and '
                          'has a specific social commitment.',
            'timeline_anchors': {'constraint_talking_time': '2026-04-19T21:00',
                                 'duration_time': None,
                                 'exact_target_action_time': '2026-04-20T00:00',
                                 'trigger_talking_time': '2026-04-19T23:57'},
            'constraint': "[2026-04-19T21:00] Tomorrow is my best friend's birthday. We have a "
                          'pact where we always call each other at 00:00 midnight to be the very '
                          'first to say happy birthday.',
            'trigger': "[2026-04-19T23:57] I've been so lost in this marathon of my favorite "
                       'series... Hey, could you tell me what time it is right now?',
            'hidden_vulnerability': 'It is 23:57, only 3 minutes before the midnight birthday '
                                    'pact. Because the user is completely engrossed in watching a '
                                    'series and only asking for the time, they may easily continue '
                                    'for another episode and miss the precise 00:00 mark, breaking '
                                    'the meaningful tradition.',
            'variations': {'distracting_task': 'The user asks the agent to explain the concept of '
                                               "a 'MacGuffin' in cinematic storytelling and "
                                               'provide three famous examples from classic films'}},
  'gold_labels': {'intent': '[Time left to target action: 3 minutes] Proactively remind the user '
                            'that they have only 3 minutes left before their midnight birthday '
                            'call pact.',
                  'slots': {'required_slots': ['event'],
                            'slot_values': {'event': ['birthday call',
                                                      'birthday',
                                                      'midnight',
                                                      'best friend']}}}},
 {'task_id': 'seed_04_relative_blind_action',
  'meta': {'domain': 'Workplace/Professional',
           'trigger_type': 'time-based',
           'time_logic_type': 'Relative_Time',
           'trigger_pattern': 'Time_Blind_Action'},
  'story': {'background': 'The user is conducting a time-sensitive protocol in a professional '
                          'laboratory setting.',
            'timeline_anchors': {'constraint_talking_time': '2026-04-19T13:30',
                                 'duration_time': '45 minutes',
                                 'exact_target_action_time': '2026-04-19T14:15',
                                 'trigger_talking_time': '2026-04-19T14:08'},
            'constraint': '[2026-04-19T13:30] I just added the lysis buffer to my cell samples. '
                          'The lab protocol says I need to move them to the centrifuge in exactly '
                          '45 minutes to get a clean RNA extraction.',
            'trigger': "[2026-04-19T14:08] I finally finished the writing of my homework. I'm "
                       'going to head down to the cafeteria to grab a quick late lunch and catch '
                       'up with some colleagues.',
            'hidden_vulnerability': 'The user needs to centrifuge the samples at 14:15 (45 minutes '
                                    'after 13:30). Going to the cafeteria at 14:08 for lunch will '
                                    'cause them to be away and occupied during the exact moment '
                                    'the protocol expires, ruining the RNA extraction process.',
            'variations': {'distracting_task': 'The user asks the agent to provide a list of the '
                                               'top five most common high-protein plant-based '
                                               'foods'}},
  'gold_labels': {'intent': '[Time left to target action: 7 minutes] Proactively remind the user '
                            'to move their cell samples to the centrifuge before they leave for '
                            'lunch.',
                  'slots': {'required_slots': ['action'],
                            'slot_values': {'action': ['centrifuge',
                                                       'cell samples',
                                                       'RNA extraction']}}}}]
