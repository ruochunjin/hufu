"""Golden-250 corpus: golden100 plus 150 cases aimed at contradiction surfaces.

Semantics pinned by the lab (2026-10-04): the tool certifies *structural*
consistency — a document must cover the query's critical content, and for
declarative claims an NLI polarity gate catches value/negation conflicts.
Answer-correctness beyond coverage is deliberately out of scope.

Families added on top of golden100:
- M: declarative conflict pairs (numbers/dates/places/persons/negation/
  antonym/role-reversal/temporal-order/quantity-direction) — the core
  "contradiction statements" channel, is_question=False;
- N: question surface extensions (how-manner, why, whose, polar yes-no,
  multi-constraint);
- O: multi-sentence documents with distractors and split answers;
- P: declarative support pairs (paraphrase/synonym/voice/restructuring);
- Q: ContraDoc-style conflict taxonomy (factual/opinion/emotion/content/
  temporal/numeric/scope);
- R: noise and boundary shapes (long sentences, lists, abbreviations,
  unit variants, mixed case).
"""

from __future__ import annotations

from tests.e2e.golden100 import CASES as BASE_CASES
from tests.e2e.golden100 import GoldenCase

NEW_CASES: list[GoldenCase] = [
    # ------------------------------------------- M. declarative conflicts (50)
    # -- numeric value conflicts (expect contradiction via NLI polarity gate)
    GoldenCase("M01", "The reservoir holds 3.2 million cubic meters of water.", "The reservoir holds 3.3 million cubic meters of water.", False, "m-numeric", is_question=False),
    GoldenCase("M02", "The bridge is 140 meters long.", "The bridge is 410 meters long.", False, "m-numeric", is_question=False),
    GoldenCase("M03", "The company employs 1200 people.", "The company employs 2100 people.", False, "m-numeric", is_question=False),
    GoldenCase("M04", "The stadium seats 65000 fans.", "The stadium seats 56000 fans.", False, "m-numeric", is_question=False),
    GoldenCase("M05", "The flight takes 3 hours.", "The flight takes 8 hours.", False, "m-numeric", is_question=False),
    GoldenCase("M06", "The treaty was signed in 1919.", "The treaty was signed in 1918.", False, "m-date", is_question=False),
    GoldenCase("M07", "The company went public in 2019.", "The company went public in 2021.", False, "m-date", is_question=False),
    GoldenCase("M08", "The bridge opened in May 1883.", "The bridge opened in June 1883.", False, "m-date", is_question=False),
    GoldenCase("M09", "The treaty was signed on 11 November 1918.", "The treaty was signed on 12 November 1918.", False, "m-date", is_question=False),
    GoldenCase("M10", "The report was published in March.", "The report was published in April.", False, "m-date", is_question=False),
    # -- place/person value conflicts
    GoldenCase("M11", "Ada Lovelace was born in London.", "Ada Lovelace was born in Paris.", False, "m-place", is_question=False),
    GoldenCase("M12", "The headquarters is in Austin, Texas.", "The headquarters is in Boston, Massachusetts.", False, "m-place", is_question=False),
    GoldenCase("M13", "The cathedral was built in Chartres.", "The cathedral was built in Reims.", False, "m-place", is_question=False),
    GoldenCase("M14", "Jane Austen wrote the novel Emma.", "Charlotte Brontë wrote the novel Emma.", False, "m-person", is_question=False),
    GoldenCase("M15", "Alexander Fleming discovered penicillin.", "Alexander Fleming discovered insulin.", False, "m-entity", is_question=False),
    GoldenCase("M16", "The CEO of the company is Mary Chen.", "The CEO of the company is John Smith.", False, "m-person", is_question=False),
    # -- negation and antonymy
    GoldenCase("M17", "The drug is approved for children.", "The drug is not approved for children.", False, "m-negation", is_question=False),
    GoldenCase("M18", "The team won the championship.", "The team lost the championship final.", False, "m-antonym", is_question=False),
    GoldenCase("M19", "The museum is open on Mondays.", "The museum is closed on Mondays.", False, "m-antonym", is_question=False),
    GoldenCase("M20", "The stock price rose sharply.", "The stock price fell sharply.", False, "m-antonym", is_question=False),
    GoldenCase("M21", "The project was a success.", "The project was a failure.", False, "m-antonym", is_question=False),
    GoldenCase("M22", "The river flows north.", "The river flows south.", None, "m-antonym", is_question=False, note="observation: direction antonym, NLI may say neutral"),
    GoldenCase("M23", "The machine increases the pressure.", "The machine decreases the pressure.", False, "m-antonym", is_question=False),
    GoldenCase("M24", "The senator supported the bill.", "The senator opposed the bill.", False, "m-antonym", is_question=False),
    # -- role reversal
    GoldenCase("M25", "The collector sold the painting to the museum.", "The museum sold the painting to the collector.", False, "m-role", is_question=False, note="clear structural role swap"),
    GoldenCase("M26", "The startup hired the engineer.", "The engineer hired the startup.", False, "m-role", is_question=False),
    GoldenCase("M27", "The dog bit the mailman.", "The mailman bit the dog.", False, "m-role", is_question=False),
    GoldenCase("M28", "The teacher praised the student.", "The student praised the teacher.", None, "m-role", is_question=False, note="observation: plausible-sounding reversal, NLI uncertain"),
    # -- temporal ordering and quantity direction
    GoldenCase("M29", "The storm came before the flood.", "The flood came before the storm.", None, "m-temporal", is_question=False, note="observation: order swap, NLI may miss"),
    GoldenCase("M30", "The renovation started after the fire.", "The renovation started before the fire.", False, "m-temporal", is_question=False),
    GoldenCase("M31", "The town has more than 12000 residents.", "The town has fewer than 8000 residents.", False, "m-quantity", is_question=False),
    GoldenCase("M32", "The team scored over 90 points.", "The team scored under 60 points.", False, "m-quantity", is_question=False),
    GoldenCase("M33", "Production doubled in a decade.", "Production halved in a decade.", False, "m-quantity", is_question=False),
    # -- paraphrase / support pairs (expect consistent)
    GoldenCase("M34", "The reservoir holds 3.2 million cubic meters of water.", "The reservoir holds 3.2 million cubic meters of water.", True, "m-support", is_question=False, note="identity control"),
    GoldenCase("M35", "Ada Lovelace was born in London in 1815.", "Ada Lovelace was born in London in 1815.", True, "m-support", is_question=False),
    GoldenCase("M36", "The company employs 1200 people.", "The company has 1200 employees.", True, "m-support", is_question=False, note="synonym paraphrase"),
    GoldenCase("M37", "The treaty was signed in 1918.", "The treaty was signed in 1918 after long negotiations.", True, "m-support", is_question=False, note="elaborated"),
    GoldenCase("M38", "The striker scored 23 goals.", "The striker netted 23 goals.", True, "m-support", is_question=False, note="synonym verb"),
    GoldenCase("M39", "The bridge was opened in 1883.", "The bridge opened to traffic in 1883.", True, "m-support", is_question=False),
    GoldenCase("M40", "Mary Chen is the CEO.", "The CEO is Mary Chen.", True, "m-support", is_question=False, note="copula inversion"),
    # -- declarative supports with pronoun chains
    GoldenCase("M41", "Ernest Hemingway wrote The Old Man and the Sea. He was born in Oak Park.", "Ernest Hemingway wrote The Old Man and the Sea. He was born in Oak Park, Illinois, in 1899.", True, "m-support", is_question=False, note="self-prefix vs elaborated"),
    GoldenCase("M42", "The violinist was born in Salzburg. She grew up in Vienna.", "The violinist was born in Salzburg. She grew up in Vienna and debuted at fifteen.", True, "m-support", is_question=False),
    # -- more number forms in conflicts
    GoldenCase("M43", "The town has 12,000 residents.", "The town has 21,000 residents.", False, "m-numeric", is_question=False, note="comma numbers"),
    GoldenCase("M44", "The dose is 0.5 milligrams.", "The dose is 5 milligrams.", False, "m-numeric", is_question=False, note="decimal shift"),
    GoldenCase("M45", "The share rose by 10 percent.", "The share rose by 20 percent.", False, "m-numeric", is_question=False),
    GoldenCase("M46", "The marathon is 42.195 kilometers.", "The marathon is 42195 meters.", True, "m-support", is_question=False, note="fixed: numeric rule compares after unit conversion"),
    GoldenCase("M47", "The marathon is 42.195 kilometers.", "The marathon is 42195 kilometers.", False, "m-numeric", is_question=False, note="unit-scale conflict"),
    GoldenCase("M48", "The probe landed on Mars in 2021.", "The probe landed on Venus in 2021.", False, "m-place", is_question=False),
    GoldenCase("M49", "The chef uses olive oil.", "The chef uses motor oil.", False, "m-entity", is_question=False, note="observation: absurd substitution, NLI uncertain"),
    GoldenCase("M50", "The library was built in 1997 and renovated in 2015.", "The library was built in 2015 and renovated in 1997.", False, "m-temporal", is_question=False, note="two-date swap"),
    # ------------------------------------------- N. question surface extensions (30)
    GoldenCase("N01", "How did the crew escape the fire?", "The crew escaped the fire through the aft hatch.", None, "how", note="observation: manner answer"),
    GoldenCase("N02", "How did the crew escape the fire?", "The fire destroyed the lower deck.", False, "how"),
    GoldenCase("N03", "Why did the flight cancel?", "The flight was canceled because of the storm.", None, "why", note="observation: reason answer"),
    GoldenCase("N04", "Why did the flight cancel?", "The flight departed two hours late.", True, "why", note="coverage-semantics: REASON is a production wildcard (any non-punct)"),
    GoldenCase("N05", "Whose signature is on the treaty?", "The treaty bears the signature of the ambassador.", None, "whose", note="observation: possessive answer"),
    GoldenCase("N06", "Whose book is on the table?", "The book on the table belongs to Maria.", True, "whose"),
    GoldenCase("N07", "Is the museum open on Mondays?", "The museum opens every day including Mondays.", True, "polar"),
    GoldenCase("N08", "Is the museum open on Mondays?", "The museum is closed on Mondays.", True, "polar", note="coverage-semantics: doc answers (negatively) with same critical vertices"),
    GoldenCase("N09", "Did the team win the championship?", "The team lost the championship final.", True, "polar", note="coverage-semantics: doc answers (negatively)"),
    GoldenCase("N10", "Did the flight depart on time?", "The flight departed two hours late.", None, "polar", note="idiom: on-time time vertex types EVENT and has no data counterpart; deep idiom handling"),
    GoldenCase("N11", "Which city hosted the 2016 Olympics?", "Rio de Janeiro hosted the 2016 Olympics.", True, "multi"),
    GoldenCase("N12", "Which city hosted the 2016 Olympics?", "The 2016 Olympics featured 28 sports.", False, "multi"),
    GoldenCase("N13", "Which planet did the probe reach in 2021?", "The probe reached Mars in 2021.", True, "multi"),
    GoldenCase("N14", "In which year did the company go public?", "The company went public in 2019.", True, "multi"),
    GoldenCase("N15", "In which year did the company go public?", "The company announced its IPO plans.", True, "multi", note="coverage-semantics: WHICH wildcard (production parity: any entity/noun)"),
    GoldenCase("N16", "Which two cities does the bridge connect?", "The bridge connects Brooklyn and Manhattan.", None, "multi", note="observation: two-part answer"),
    GoldenCase("N17", "How long is the marathon race?", "The marathon race is 42195 meters long.", True, "number"),
    GoldenCase("N18", "How long is the marathon race?", "The marathon race takes hours to finish.", False, "number"),
    GoldenCase("N19", "How far is the airport from the city?", "The airport is 40 kilometers from the city.", True, "number"),
    GoldenCase("N20", "How far is the airport from the city?", "The airport serves the whole metropolitan area.", False, "number"),
    GoldenCase("N21", "How often does the ferry run?", "The ferry runs every thirty minutes.", None, "how", note="observation: frequency answer"),
    GoldenCase("N22", "How old is the cathedral?", "The cathedral is 800 years old.", True, "number"),
    GoldenCase("N23", "How old is the cathedral?", "The cathedral dominates the old town.", True, "number", note="coverage-semantics: how-old marks ATTRIBUTE placeholder; doc old(ADJ) matches (production parity)"),
    GoldenCase("N24", "What kind of fuel does the rocket use?", "The rocket uses liquid hydrogen.", None, "what", note="observation: kind answer"),
    GoldenCase("N25", "What happened to the expedition in 1912?", "The expedition reached the pole in 1912.", None, "what", note="observation: open what question"),
    GoldenCase("N26", "Who did the committee appoint as director?", "The committee appointed Dr. Reyes as director.", True, "who"),
    GoldenCase("N27", "Who did the committee appoint as director?", "The committee announced the new budget.", False, "who"),
    GoldenCase("N28", "Where did the ship sink after leaving Liverpool?", "The ship sank in the North Atlantic after leaving Liverpool.", True, "where"),
    GoldenCase("N29", "Where did the ship sink after leaving Liverpool?", "The ship left Liverpool on Tuesday.", False, "where"),
    GoldenCase("N30", "When exactly did the library reopen after the fire?", "The library reopened in March 2016 after the fire.", True, "when"),
    # ------------------------------------------- O. multi-sentence / distractor (20)
    GoldenCase("O01", "How many goals did the striker score?", "The season started slowly. The midfielder scored 4 goals in autumn. The striker scored 23 goals by May.", True, "multi-sent"),
    GoldenCase("O02", "When did the war end?", "The war began in 1939 with an invasion. Millions fled their homes across the continent. The war ended in 1945.", True, "multi-sent"),
    GoldenCase("O03", "Where was the violinist born?", "The violinist grew up in Vienna and debuted at fifteen. She performed in Salzburg for years. She was born in Salzburg in 1981.", True, "multi-sent", note="born city repeated later"),
    GoldenCase("O04", "Where was the violinist born?", "The violinist grew up in Vienna and debuted at fifteen. She performed in Salzburg for years.", True, "multi-sent", note="coverage-semantics: GPE targets present for ?Where"),
    GoldenCase("O05", "Who painted the chapel ceiling?", "The chapel took a century to complete. Many artists contributed frescoes. The ceiling of the chapel was painted by Michelangelo.", True, "multi-sent"),
    GoldenCase("O06", "How many residents does the town have?", "The town was founded by miners in 1880. It survived two fires and a flood. The town has 12000 residents today.", True, "multi-sent"),
    GoldenCase("O07", "When was the company founded?", "The company launched its first product in 2005. The company was founded in 1998 in a garage.", True, "multi-sent", note="earlier fact in second sentence"),
    GoldenCase("O08", "Who wrote the novel about the whale?", "The novel about the whale was published in 1851. Herman Melville wrote the novel about the whale.", True, "multi-sent"),
    GoldenCase("O09", "How many passengers were on the ship?", "The ship left Liverpool on Tuesday. It carried mail and cargo. 1322 passengers were on the ship.", None, "multi-sent", note="observation: bare number sentence"),
    GoldenCase("O10", "Where did the expedition winter?", "The expedition left Oslo in June. The expedition wintered at Cape Evans.", True, "multi-sent"),
    GoldenCase("O11", "Who discovered the tomb?", "The tomb lay hidden for three thousand years. Howard Carter discovered the tomb in 1922.", True, "multi-sent"),
    GoldenCase("O12", "When did the reactor shut down?", "The reactor came online in 1983. The reactor shut down in 1986.", True, "multi-sent"),
    GoldenCase("O13", "How many satellites did the rocket carry?", "The rocket lifted off at dawn. It carried 12 satellites into orbit.", True, "multi-sent"),
    GoldenCase("O14", "Who directed the film?", "The film won four awards at the festival. Denis Villeneuve directed the film.", True, "multi-sent"),
    GoldenCase("O15", "When did the volcano last erupt?", "The volcano towers over the island. It last erupted in 1707.", True, "multi-sent"),
    GoldenCase("O16", "Where was the manuscript discovered?", "Scholars searched monasteries for decades. The manuscript was discovered in a cave near Qumran.", True, "multi-sent"),
    GoldenCase("O17", "How many Oscars did the composer win?", "The composer wrote forty film scores. She won 3 Oscars for her work.", True, "multi-sent"),
    GoldenCase("O18", "Who invented the telephone?", "Communication changed forever in 1876. Alexander Graham Bell invented the telephone.", True, "multi-sent"),
    GoldenCase("O19", "When did the subway first open?", "The city grew rapidly in the 1800s. The subway first opened in 1904.", None, "multi-sent", note="builder asymmetry: query first is a critical ordinal vertex; data merges first+opened into an unmatchable verb predicate"),
    GoldenCase("O20", "Where is the observatory located?", "Astronomers needed dry, dark skies. The observatory is located in the Atacama Desert.", True, "multi-sent"),
    # ------------------------------------------- P. declarative supports (20)
    GoldenCase("P01", "The company was founded in 1998.", "The company was founded in 1998 in a small garage.", True, "p-support", is_question=False, note="elaborated"),
    GoldenCase("P02", "Herman Melville wrote the whale novel.", "The novel about the whale was written by Herman Melville.", True, "p-support", is_question=False, note="active to passive"),
    GoldenCase("P03", "The ceiling was painted by Michelangelo.", "Michelangelo painted the ceiling.", True, "p-support", is_question=False, note="passive to active"),
    GoldenCase("P04", "The rocket carried 12 satellites.", "Twelve satellites rode the rocket into orbit.", True, "p-support", is_question=False, note="restructured"),
    GoldenCase("P05", "The striker scored 23 goals this season.", "This season, the striker scored 23 goals.", True, "p-support", is_question=False, note="fronted adverbial"),
    GoldenCase("P06", "The town has 12000 residents.", "The town is home to 12000 residents.", True, "p-support", is_question=False, note="synonym predicate"),
    GoldenCase("P07", "The flight was canceled because of the storm.", "Because of the storm, the flight was canceled.", True, "p-support", is_question=False),
    GoldenCase("P08", "The probe reached Mars in 2021.", "In 2021, the probe reached Mars.", True, "p-support", is_question=False),
    GoldenCase("P09", "The violinist was born in Salzburg.", "Salzburg is the birthplace of the violinist.", True, "p-support", is_question=False),
    GoldenCase("P10", "The reactor shut down in 1986.", "The reactor was shut down in 1986.", True, "p-support", is_question=False, note="ergative alternation"),
    GoldenCase("P11", "The CEO is Mary Chen.", "Mary Chen serves as CEO.", True, "p-support", is_question=False),
    GoldenCase("P12", "The observatory sits in the Atacama Desert.", "The observatory is located in the Atacama Desert.", True, "p-support", is_question=False),
    GoldenCase("P13", "The tomb lay hidden for three thousand years.", "For three thousand years, the tomb lay hidden.", True, "p-support", is_question=False),
    GoldenCase("P14", "The ship carried 1322 passengers.", "1322 passengers were aboard the ship.", True, "p-support", is_question=False),
    GoldenCase("P15", "The cathedral is 800 years old.", "The cathedral was built 800 years ago.", True, "p-support", is_question=False, note="age vs built-ago"),
    GoldenCase("P16", "The airport is 40 kilometers from the city.", "The airport lies 40 kilometers away from the city.", True, "p-support", is_question=False),
    GoldenCase("P17", "The company employs 1200 people.", "The company has a workforce of 1200 people.", True, "p-support", is_question=False),
    GoldenCase("P18", "The film won four awards.", "Four awards went to the film.", True, "p-support", is_question=False),
    GoldenCase("P19", "The manuscript was found in a cave near Qumran.", "Near Qumran, in a cave, the manuscript was found.", True, "p-support", is_question=False),
    GoldenCase("P20", "The subway first opened in 1904.", "The subway opened for the first time in 1904.", True, "p-support", is_question=False),
    # ------------------------------------------- Q. ContraDoc-style taxonomy (15)
    GoldenCase("Q01", "Holmes grows angry when Watson touches his things.", "Holmes does not mind his things being touched.", False, "q-opinion", is_question=False, note="Perspective/Opinion"),
    GoldenCase("Q02", "The Traveller is relieved that Weena was lost in the fire.", "The Traveller plans to take Weena back to his own time.", None, "q-emotion", is_question=False, note="limitation: emotion inference beyond sentence NLI"),
    GoldenCase("Q03", "Smith sees the little ivory box that he had sent to Holmes.", "Smith never sent any box to Holmes.", False, "q-content", is_question=False, note="Content"),
    GoldenCase("Q04", "Yorick returns to Calais after staying in Paris for a few days.", "Yorick never visited Paris on his journey.", False, "q-temporal", is_question=False, note="Temporal"),
    GoldenCase("Q05", "The regiment lost 300 men at the ridge.", "The regiment lost 3000 men at the ridge.", False, "q-numeric", is_question=False, note="Numeric"),
    GoldenCase("Q06", "The entire village was evacuated before the flood.", "Only the village school was evacuated before the flood.", False, "q-scope", is_question=False, note="Scope"),
    GoldenCase("Q07", "All of the expedition's dogs survived the winter.", "None of the expedition's dogs survived the winter.", False, "q-scope", is_question=False),
    GoldenCase("Q08", "The queen attended the ceremony in person.", "The queen did not attend the ceremony.", False, "q-content", is_question=False),
    GoldenCase("Q09", "The novel ends with the detective retiring to the countryside.", "The novel ends with the detective being murdered in the city.", False, "q-content", is_question=False),
    GoldenCase("Q10", "The witness described the suspect as tall and bearded.", "The witness described the suspect as short and clean-shaven.", False, "q-content", is_question=False),
    GoldenCase("Q11", "The locals welcomed the expedition warmly.", "The locals greeted the expedition with hostility.", False, "q-opinion", is_question=False),
    GoldenCase("Q12", "The miners struck gold in the second shaft.", "The miners found nothing in the second shaft.", False, "q-content", is_question=False),
    GoldenCase("Q13", "The king died peacefully in his sleep.", "The king was assassinated in his bed.", None, "q-content", is_question=False, note="limitation: manner-of-death antonymy, NLI uncertain"),
    GoldenCase("Q14", "The river froze solid that winter.", "The river never froze that winter.", False, "q-negation", is_question=False),
    GoldenCase("Q15", "The captain praised the crew's courage.", "The captain blamed the crew for the disaster.", False, "q-opinion", is_question=False),
    # ------------------------------------------- R. noise and boundary (15)
    GoldenCase("R01", "Where was the data center built?", "The data center was built in Oslo.", True, "noise", note="simple control"),
    GoldenCase("R02", "When did Version 2.0 ship?", "Version 2.0 shipped in September.", None, "noise", note="sm parse artifact: ship parsed as noun, merged 2.0 ship compound; trf check"),
    GoldenCase("R03", "How many CVEs were patched in the update?", "The update patched 14 CVEs.", True, "noise", note="abbreviation"),
    GoldenCase("R04", "Who won the F1 championship in 2020?", "Lewis Hamilton won the F1 championship in 2020.", True, "noise", note="F1 as racing"),
    GoldenCase("R05", "Where is the 5G base station located?", "The 5G base station is located on the hill.", True, "noise", note="alphanumeric token"),
    GoldenCase("R06", "How many COVID-19 cases were reported?", "The city reported 320 COVID-19 cases.", True, "noise", note="hyphenated token"),
    GoldenCase("R07", "When did the U.S. enter the war?", "The U.S. entered the war in 1941.", True, "noise", note="dotted abbreviation"),
    GoldenCase("R08", "Where did the G7 summit take place?", "The G7 summit took place in Cornwall.", True, "noise"),
    GoldenCase("R09", "How many episodes are in the mini-series?", "The mini-series has 6 episodes.", True, "noise"),
    GoldenCase("R10", "When was the iPhone released?", "The iPhone was released in 2007.", True, "noise", note="camel case brand"),
    GoldenCase("R11", "Who wrote the novel 1984?", "George Orwell wrote the novel 1984.", True, "noise", note="numeric title"),
    GoldenCase("R12", "How many people can the Airbus A380 carry?", "The Airbus A380 can carry 853 passengers.", True, "noise", note="model number in name"),
    GoldenCase("R13", "Where is Mount Everest?", "Mount Everest is in the Himalayas.", None, "noise", note="observation: region not GPE"),
    GoldenCase("R14", "When did Apollo 11 land on the moon?", "Apollo 11 landed on the moon in 1969.", True, "noise"),
    GoldenCase("R15", "How many bones are in the human hand?", "There are 27 bones in the human hand.", True, "noise"),
]

CASES: list[GoldenCase] = list(BASE_CASES) + NEW_CASES
assert len(CASES) == 250, f"corpus should hold exactly 250 cases, got {len(CASES)}"
_ids = [case.id for case in CASES]
assert len(_ids) == len(set(_ids)), "duplicate case ids across corpus"
