"""Few-shot user-only dialogues for Temporal Grounding Stage 2.

All four seeds have dialogues: Stage 2 selects the matching
(time_logic_type × trigger_pattern) example as in-context learning.
"""

SEED_USER_DIALOGUES = {'seed_01_absolute_blind_action': [{'turn_id': 1,
                                    'time': '2026-04-19T08:15',
                                    'function': 'thematic_connection',
                                    'text': "Good morning! I'm planning to get my best friend "
                                            'a really meaningful gift this year. Can you '
                                            'suggest a few unique, personalized gift ideas for '
                                            'someone who loves analog photography?'},
                                   {'turn_id': 2,
                                    'time': '2026-04-19T08:35',
                                    'function': 'thematic_connection',
                                    'text': 'Those are great ideas. Could you also draft a '
                                            'short, heartfelt message I can write on the '
                                            'inside of the vintage camera case?'},
                                   {'turn_id': 3,
                                    'time': '2026-04-19T09:00',
                                    'function': 'constraint',
                                    'text': "Tomorrow is my best friend's birthday. We have a "
                                            'pact where we always call each other at 00:00 '
                                            'midnight to be the very first to say happy '
                                            'birthday.'},
                                   {'turn_id': 4,
                                    'time': '2026-04-19T17:45',
                                    'function': 'thematic_connection',
                                    'text': 'I feel so tired. I just finished a long meeting '
                                            'with my team about the new project promotion '
                                            'strategy. Can you recommend some suggestions for '
                                            "improving our product's social media presence to "
                                            'attract more users?'},
                                   {'turn_id': 5,
                                    'time': '2026-04-19T21:30',
                                    'function': 'thematic_connection',
                                    'text': "It's been a long day. I'm finally relaxing at "
                                            'home now. Could you recommend some ambient '
                                            'instrumental music that is good for unwinding?'},
                                   {'turn_id': 6,
                                    'time': '2026-04-19T22:45',
                                    'function': 'thematic_connection',
                                    'text': 'I was reading an article about sleep cycles '
                                            'earlier. Can you quickly summarize the difference '
                                            'between REM sleep and deep sleep?'},
                                   {'turn_id': 7,
                                    'time': '2026-04-19T23:50',
                                    'function': 'trigger',
                                    'text': "I'm feeling unexpectedly exhausted today. I'm "
                                            'going to put on my sleep mask, and crash for the '
                                            'night.'}],
 'seed_02_relative_time_query': [{'turn_id': 1,
                                  'time': '2026-04-19T13:10',
                                  'function': 'thematic_connection',
                                  'text': "OMG, I have so many things to do these days! I'm in "
                                          'the biology lab doing some prep work, but I realize '
                                          'I also need to finish my math homework and submit '
                                          'it today. Can you remind me of the formula for '
                                          'finding the derivative of a trigonometric '
                                          'function?'},
                                 {'turn_id': 2,
                                  'time': '2026-04-19T13:22',
                                  'function': 'thematic_connection',
                                  'text': 'Thanks. How about the chain rule? Could you walk me '
                                          'through a step-by-step example of applying the '
                                          'chain rule to a composite function?'},
                                 {'turn_id': 3,
                                  'time': '2026-04-19T13:30',
                                  'function': 'constraint',
                                  'text': 'Ok, I just added the lysis buffer to my cell '
                                          'samples. The lab protocol says I need to move them '
                                          'to the centrifuge in exactly 45 minutes to get a '
                                          'clean RNA extraction.'},
                                 {'turn_id': 4,
                                  'time': '2026-04-19T13:36',
                                  'function': 'thematic_connection',
                                  'text': 'Sometimes my experimental results vary a lot even '
                                          'when I follow the same protocol. What are common '
                                          'sources of variability in wet lab experiments?'},
                                 {'turn_id': 5,
                                  'time': '2026-04-19T13:45',
                                  'function': 'thematic_connection',
                                  'text': "Alright, let's get back to the math. Can you "
                                          'explain the concept of eigenvalues and eigenvectors '
                                          'in linear algebra? I need to understand it for the '
                                          'next section.'},
                                 {'turn_id': 6,
                                  'time': '2026-04-19T14:00',
                                  'function': 'thematic_connection',
                                  'text': 'That makes sense. Can you show me a concrete 2x2 '
                                          'matrix example and calculate its eigenvalues so I '
                                          'can see the process in action?'},
                                 {'turn_id': 7,
                                  'time': '2026-04-19T14:12',
                                  'function': 'trigger',
                                  'text': 'What time is it? I got so deeply focused on my math '
                                          'homework.'}],
 'seed_03_absolute_time_query': [{'turn_id': 1,
                                  'time': '2026-04-19T19:15',
                                  'function': 'thematic_connection',
                                  'text': 'I had three meetings today and feel exhausted. My '
                                          'manager pushed a lot of work on me last minute, and '
                                          'I just finished it. Please help me to draft an '
                                          'email to my manager. I want to express that I have '
                                          'completed the tasks, and would like to book a '
                                          'meeting to discuss the project timeline tomorrow.'},
                                 {'turn_id': 2,
                                  'time': '2026-04-19T20:15',
                                  'function': 'thematic_connection',
                                  'text': "I finally arrived home. I'm planning to cook a nice "
                                          'dinner for myself. Can you suggest a simple but '
                                          'delicious recipe that I can make with chicken '
                                          'breast, broccoli, and rice?'},
                                 {'turn_id': 3,
                                  'time': '2026-04-19T21:00',
                                  'function': 'constraint',
                                  'text': "Tomorrow is my best friend's birthday. We have a "
                                          'pact where we always call each other at 00:00 '
                                          'midnight to be the very first to say happy '
                                          'birthday.'},
                                 {'turn_id': 4,
                                  'time': '2026-04-19T21:07',
                                  'function': 'thematic_connection',
                                  'text': "I want to relax and watch a movie tonight. I'm "
                                          'thinking about starting a classic thriller series '
                                          'marathon. Can you recommend a good thriller series '
                                          'to start with, and maybe give me a brief overview '
                                          'of the plot and main characters?'},
                                 {'turn_id': 5,
                                  'time': '2026-04-19T22:15',
                                  'function': 'thematic_connection',
                                  'text': 'This series is amazing! Speaking of cinematic '
                                          'storytelling, can you explain the concept of a '
                                          "'MacGuffin' and why directors use it to drive the "
                                          'plot forward?'},
                                 {'turn_id': 6,
                                  'time': '2026-04-19T23:10',
                                  'function': 'thematic_connection',
                                  'text': 'What about lighting? How do cinematographers use '
                                          'shadows and low-key lighting to build tension in '
                                          'scenes where the main character is isolated?'},
                                 {'turn_id': 7,
                                  'time': '2026-04-19T23:57',
                                  'function': 'trigger',
                                  'text': "I've been so lost in this marathon of my favorite "
                                          'series... Hey, could you tell me what time it is '
                                          'right now?'}],
 'seed_04_relative_blind_action': [{'turn_id': 1,
                                    'time': '2026-04-19T13:05',
                                    'function': 'thematic_connection',
                                    'text': "I'm in the professional laboratory setting up "
                                            "some equipment, and while I wait, I'm working on "
                                            'a formal review paper. What are the key '
                                            'differences between a systematic review and a '
                                            'meta-analysis?'},
                                   {'turn_id': 2,
                                    'time': '2026-04-19T13:20',
                                    'function': 'thematic_connection',
                                    'text': 'Could you provide a few strong transition phrases '
                                            'to link a paragraph discussing methodology '
                                            'limitations to a paragraph introducing future '
                                            'research directions?'},
                                   {'turn_id': 3,
                                    'time': '2026-04-19T13:30',
                                    'function': 'constraint',
                                    'text': 'I just added the lysis buffer to my cell samples. '
                                            'The lab protocol says I need to move them to the '
                                            'centrifuge in exactly 45 minutes to get a clean '
                                            'RNA extraction.'},
                                   {'turn_id': 4,
                                    'time': '2026-04-19T13:45',
                                    'function': 'thematic_connection',
                                    'text': "Back to the paper. What's the standard format for "
                                            'citing an unpublished clinical trial in APA 7th '
                                            'edition?'},
                                   {'turn_id': 5,
                                    'time': '2026-04-19T13:58',
                                    'function': 'thematic_connection',
                                    'text': 'Can you give me a checklist of things to '
                                            'double-check in an abstract before submitting a '
                                            'paper to a high-impact biology journal?'},
                                   {'turn_id': 6,
                                    'time': '2026-04-19T14:08',
                                    'function': 'trigger',
                                    'text': 'I finally finished the writing of my homework. '
                                            "I'm going to head down to the cafeteria to grab a "
                                            'quick late lunch and catch up with some '
                                            'colleagues.'}]}
