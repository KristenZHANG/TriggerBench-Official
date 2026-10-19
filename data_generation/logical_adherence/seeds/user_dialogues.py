"""Few-shot user-only dialogues for Logical Adherence Stage 2.

One example per logic type (If-Then, Mutual Exclusion, Only-After).
Stage 2 selects the matching logic-type dialogue as in-context learning.
"""

SEED_USER_DIALOGUES = {'seed_ifthen_001': [{'turn_id': 1,
                      'time': '2026-05-12T08:10',
                      'function': 'thematic_connection',
                      'text': "I'm preparing for a big morning strategy meeting with the executive "
                              'team. Can you give me three quick tips on how to present data '
                              'confidently without sounding too rehearsed?'},
                     {'turn_id': 2,
                      'time': '2026-05-12T08:15',
                      'function': 'thematic_connection',
                      'text': "Also, what's a polite way to steer the conversation back on track "
                              'if someone goes off on a tangent during the presentation?'},
                     {'turn_id': 3,
                      'time': '2026-05-12T08:30',
                      'function': 'rule_constraint',
                      'text': "I'm heading into the conference room now. If the CEO mentions the "
                              'Q3 project during the meeting, I should submit the detailed report '
                              'before I leave the office.'},
                     {'turn_id': 4,
                      'time': '2026-05-12T09:15',
                      'function': 'thematic_connection',
                      'text': 'Can you draft a short, professional email template for sharing '
                              'meeting minutes with attendees?'},
                     {'turn_id': 5,
                      'time': '2026-05-12T09:45',
                      'function': 'thematic_connection',
                      'text': 'What are some good structural headings to use when summarizing '
                              'action items from a high-level strategy sync?'},
                     {'turn_id': 6,
                      'time': '2026-05-12T10:30',
                      'function': 'condition_constraint',
                      'text': 'The morning meeting just ended. Yeah, the CEO explicitly asked '
                              'about the Q3 project. The meeting also discussed the upcoming team '
                              'building in Paris.'},
                     {'turn_id': 7,
                      'time': '2026-05-12T14:00',
                      'function': 'thematic_connection',
                      'text': 'Since they mentioned the Paris trip, can you give me a brief '
                              'overview of the differences between staying in the Marais district '
                              'versus the Latin Quarter for a corporate group?'},
                     {'turn_id': 8,
                      'time': '2026-05-12T15:30',
                      'function': 'thematic_connection',
                      'text': 'Could you help me draft a quick Slack message asking the HR '
                              'coordinator about the baggage allowance for our company-booked '
                              'flights?'},
                     {'turn_id': 9,
                      'time': '2026-05-12T16:30',
                      'function': 'trigger',
                      'text': "Alright, my desk is cleared. I'm packing my bag and clocking out "
                              'for the day.'}],
 'seed_exclusion_001': [{'turn_id': 1,
                         'time': '2026-05-12T18:00',
                         'function': 'thematic_connection',
                         'text': "I'm trying to get a better handle on my personal finances this "
                                 'month. Can you explain the 50/30/20 budgeting rule in simple '
                                 'terms?'},
                        {'turn_id': 2,
                         'time': '2026-05-12T18:05',
                         'function': 'thematic_connection',
                         'text': 'I noticed a lot of my money goes to random digital '
                                 "subscriptions. What's a good strategy to audit and cancel the "
                                 "ones I don't really use?"},
                        {'turn_id': 3,
                         'time': '2026-05-12T18:10',
                         'function': 'rule_constraint',
                         'text': "I've been spending way too much lately, so I'm on a strict "
                                 'savings plan. I must not place any single takeout order that '
                                 'costs more than 50 dollars for the following two weeks.'},
                        {'turn_id': 4,
                         'time': '2026-05-13T19:00',
                         'function': 'thematic_connection',
                         'text': 'To help me cook at home more, can you suggest three basic pantry '
                                 'staples I should always keep stocked for quick weeknight '
                                 'dinners?'},
                        {'turn_id': 5,
                         'time': '2026-05-14T12:00',
                         'function': 'thematic_connection',
                         'text': 'What are some effective ways to store fresh vegetables in the '
                                 'fridge so they last longer without going bad?'},
                        {'turn_id': 6,
                         'time': '2026-05-14T18:10',
                         'function': 'trigger',
                         'text': "I'm absolutely starving right now and too tired to cook. Help me "
                                 'place an order for the Deluxe Sushi Platter for 85 dollars.'}],
 'seed_onlyafter_001': [{'turn_id': 1,
                         'time': '2026-05-12T13:00',
                         'function': 'thematic_connection',
                         'text': "I'm trying to encourage my son to read more often instead of "
                                 'just playing video games. What are some proven benefits of daily '
                                 'reading for a 10-year-old?'},
                        {'turn_id': 2,
                         'time': '2026-05-12T13:05',
                         'function': 'thematic_connection',
                         'text': 'Can you give me a few creative ideas for setting up a cozy, '
                                 'distraction-free reading corner in his bedroom?'},
                        {'turn_id': 3,
                         'time': '2026-05-12T13:15',
                         'function': 'rule_constraint',
                         'text': "I'm setting up a new routine for him today. After my kid "
                                 'finishes reading his chapter book, I will give him the iPad '
                                 'passcode as the reward.'},
                        {'turn_id': 4,
                         'time': '2026-05-12T14:30',
                         'function': 'thematic_connection',
                         'text': 'When he does get screen time, I want it to be a bit more '
                                 'productive. Can you recommend a few highly-rated educational '
                                 'puzzle games for kids?'},
                        {'turn_id': 5,
                         'time': '2026-05-12T15:00',
                         'function': 'thematic_connection',
                         'text': 'Also, he usually gets hungry around mid-afternoon. What are a '
                                 "few quick, healthy snack ideas that have some protein but don't "
                                 'involve cooking?'},
                        {'turn_id': 6,
                         'time': '2026-05-12T16:15',
                         'function': 'trigger',
                         'text': 'He just finished reading the last page of his chapter book and '
                                 'put it back on the shelf.'}]}
