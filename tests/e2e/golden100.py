"""Golden case corpus: ~100 diverse query–document pairs for the reference engine.

Every case pins the tool's *structural coverage* semantics: a document is
``expected=True`` (consistent) when it structurally covers the query's
content requirements, ``expected=False`` when it fails to — including when
it omits the queried entity/number/date entirely.  A document stating a
*different* but type-compatible answer still covers the query structurally
(coverage is not answer correctness); those cases carry
``note="coverage-semantics"`` to pin the distinction deliberately.

Cases with genuinely uncertain engine behaviour carry ``expected=None`` and
are report-only observations — they never fail the suite; triage promotes
them to pinned expectations once understood.

Categories: where / when / how-many / who / what-which / negation /
voice-reversal / comparison / declarative claims / distractors / unicode /
edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    id: str
    query: str
    document: str
    expected: bool | None
    category: str
    is_question: bool = True
    note: str = ""


CASES: list[GoldenCase] = [
    # ------------------------------------------------------------------ A. where (12)
    GoldenCase("A01", "Where was Ada Lovelace born?", "Ada Lovelace was born in London in 1815.", True, "where"),
    GoldenCase("A02", "Where was Ada Lovelace born?", "Ada Lovelace was an English mathematician.", None, "where", note="model-dependent: sm leaves English untyped -> contra; trf labels NORP -> consistent (production location group includes NORP). Boundary accepted as-is."),
    GoldenCase("A03", "Where did the meeting take place?", "The meeting took place in Geneva on Monday.", True, "where"),
    GoldenCase("A04", "Where did the meeting take place?", "The meeting was postponed to next Monday.", False, "where"),
    GoldenCase("A05", "Where was the cathedral built?", "The cathedral was built in Chartres.", True, "where"),
    GoldenCase("A06", "Where was the cathedral built?", "The cathedral was built in Reims.", True, "where", note="coverage-semantics: wrong place still covers"),
    GoldenCase("A07", "Where was Ada Lovelace born?", "Ada Lovelace was born in Paris.", True, "where", note="coverage-semantics: Paris vs London"),
    GoldenCase("A08", "Where is the headquarters of the company?", "The company moved its headquarters to Austin, Texas.", True, "where"),
    GoldenCase("A09", "Where is the headquarters of the company?", "The company announced record profits this year.", False, "where"),
    GoldenCase("A10", "Where did she hide the letter?", "She hid the letter under the floorboard.", None, "where", note="observation: non-GPE location phrase"),
    GoldenCase("A11", "Where was the painting found?", "The painting was found in a vault in Zurich.", True, "where"),
    GoldenCase("A12", "Where was the painting found?", "The painting vanished from the museum in 1990.", False, "where"),
    # ------------------------------------------------------------------ B. when (10)
    GoldenCase("B01", "When was the treaty signed?", "The treaty was signed on 11 November 1918.", True, "when"),
    GoldenCase("B02", "When was the treaty signed?", "The treaty ended the war in Europe.", False, "when"),
    GoldenCase("B03", "When did the company go public?", "The company went public in 2019.", True, "when"),
    GoldenCase("B04", "When did the company go public?", "The company went public in 2021.", True, "when", note="coverage-semantics: wrong year still covers"),
    GoldenCase("B05", "When did the flood happen?", "The flood happened last spring.", None, "when", note="observation: relative time phrase"),
    GoldenCase("B06", "When was the bridge opened?", "The bridge was opened on 24 May 1883.", True, "when"),
    GoldenCase("B07", "When was the bridge opened?", "The bridge connects two boroughs of the city.", False, "when"),
    GoldenCase("B08", "When did the expedition reach the pole?", "The expedition reached the pole in December 1911.", True, "when"),
    GoldenCase("B09", "When was the law passed?", "The law was passed in 1964 after long debate.", True, "when"),
    GoldenCase("B10", "When was the law passed?", "The law remains controversial today.", True, "when", note="coverage-semantics: 'today' is TEMPORAL -> covered"),
    # ------------------------------------------------------------------ C. how many / number (12)
    GoldenCase("C01", "How many meters is the marathon race?", "The marathon race is 42195 meters.", True, "number"),
    GoldenCase("C02", "How many meters is the marathon race?", "The marathon race is very long.", False, "number"),
    GoldenCase("C03", "How many planets are in the solar system?", "There are eight planets in the solar system.", None, "number", note="observation: number as word"),
    GoldenCase("C04", "How many planets are in the solar system?", "The solar system contains 8 planets.", True, "number"),
    GoldenCase("C05", "How many employees does the company have?", "The company has 1200 employees.", True, "number"),
    GoldenCase("C06", "How many employees does the company have?", "The company is hiring aggressively.", False, "number"),
    GoldenCase("C07", "How many people attended the concert?", "About 5000 people attended the concert.", True, "number"),
    GoldenCase("C08", "How many seats does the stadium have?", "The stadium has 65,000 seats.", True, "number", note="thousands separator"),
    GoldenCase("C09", "How much does the book cost?", "The book costs 25 dollars.", True, "number"),
    GoldenCase("C10", "How much does the book cost?", "The book is available in bookstores.", False, "number"),
    GoldenCase("C11", "What percentage of students passed the exam?", "92% of students passed the exam.", True, "number"),
    GoldenCase("C12", "How many goals did the striker score?", "The striker scored 23 goals this season.", True, "number", note="fixed: lexical containment hv rule for merged compounds"),
    # ------------------------------------------------------------------ D. who / person (10)
    GoldenCase("D01", "Who wrote the novel Emma?", "Jane Austen wrote the novel Emma.", True, "who"),
    GoldenCase("D02", "Who wrote the novel Emma?", "The novel Emma was published in 1815.", False, "who"),
    GoldenCase("D03", "Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928.", True, "who"),
    GoldenCase("D04", "Who discovered penicillin?", "Penicillin revolutionized medicine.", False, "who"),
    GoldenCase("D05", "Who won the Nobel Prize in Physics in 1921?", "Albert Einstein won the Nobel Prize in Physics in 1921.", True, "who"),
    GoldenCase("D06", "Who won the Nobel Prize in Physics in 1921?", "Niels Bohr won the Nobel Prize in Physics in 1922.", True, "who", note="coverage-semantics: a different person+year still covers Who/When"),
    GoldenCase("D07", "Who is the CEO of the company?", "Mary Chen is the CEO of the company.", True, "who"),
    GoldenCase("D08", "Who is the CEO of the company?", "John Smith founded the company in 1998.", True, "who", note="coverage-semantics: founder mention gives a PERSON target for ?Who"),
    GoldenCase("D09", "Who did the committee appoint?", "The committee appointed Dr. Reyes.", None, "who", note="observation: object-position person"),
    GoldenCase("D10", "Who wrote the novel Emma?", "She wrote the novel Emma.", None, "who", note="observation: unresolved pronoun without coref antecedent"),
    # ------------------------------------------------------------------ E. what / which / attribute (8)
    GoldenCase("E01", "What is the capital of France?", "The capital of France is Paris.", True, "what"),
    GoldenCase("E02", "What is the capital of France?", "France is a country in Europe.", True, "what", note="coverage-semantics+F2: capital demoted to predicate-root, ?What wildcard matches country"),
    GoldenCase("E03", "What color is the rescue helicopter?", "The rescue helicopter is bright yellow.", None, "what", note="observation: color attribute"),
    GoldenCase("E04", "What language do they speak in Brazil?", "They speak Portuguese in Brazil.", True, "what"),
    GoldenCase("E05", "Which planet is closest to the sun?", "Mercury is the planet closest to the sun.", True, "which"),
    GoldenCase("E06", "Which planet is closest to the sun?", "Venus is the hottest planet.", None, "which", note="limitation: WHICH wildcard covers any entity; semantic discrimination needs D-match tuning"),
    GoldenCase("E07", "What instrument does she play?", "She plays the violin in the orchestra.", True, "what"),
    GoldenCase("E08", "What instrument does she play?", "She rehearses every evening.", None, "what", note="limitation: WHAT wildcard covers any noun; false-consistent risk"),
    # ------------------------------------------------------------------ F. negation (6)
    GoldenCase("F01", "Did the team win the championship?", "The team did not win the championship this year.", None, "negation", note="observation: negated doc"),
    GoldenCase("F02", "Did the team win the championship?", "The team won the championship last year.", True, "negation"),
    GoldenCase("F03", "Is the drug approved?", "The drug is not approved for children.", None, "negation", note="observation: partial negation"),
    GoldenCase("F04", "Did the flight depart on time?", "The flight never departed on time last winter.", None, "negation", note="observation: never"),
    GoldenCase("F05", "Did the team win the championship?", "The team lost the final match.", None, "negation", note="limitation: antonymy/negation not modeled in coverage semantics"),
    GoldenCase("F06", "Is the museum open on Mondays?", "The museum is closed on Mondays.", True, "negation", note="coverage-semantics: open/closed both ADJ; negation is not modeled"),
    # ------------------------------------------------------------------ G. voice / role reversal (8)
    GoldenCase("G01", "Who sold the painting to the museum?", "The collector sold the painting to the museum.", True, "voice"),
    GoldenCase("G02", "Who sold the painting to the museum?", "The museum sold the painting to the collector.", None, "voice", note="role reversal: reversed roles"),
    GoldenCase("G03", "Who hired the engineer?", "The startup hired the engineer.", None, "voice", note="startup mistyped EVENT by WordNet seed; PERSON gate covers ORG but not EVENT"),
    GoldenCase("G04", "Who hired the engineer?", "The engineer hired the startup.", None, "voice", note="implausible reversal"),
    GoldenCase("G05", "Who did the dog bite?", "The dog bit the mailman.", None, "voice", note="object position"),
    GoldenCase("G06", "Who did the dog bite?", "The mailman bit the dog.", None, "voice", note="role reversal"),
    GoldenCase("G07", "Who taught the course?", "The course was taught by Professor Ito.", True, "voice", note="passive voice equivalent"),
    GoldenCase("G08", "Who taught the course?", "Professor Ito taught the course.", True, "voice"),
    # ------------------------------------------------------------------ H. comparison / quantity words (6)
    GoldenCase("H01", "How many points did the team score?", "The team scored more than 90 points.", None, "comparison", note="observation: more-than phrasing"),
    GoldenCase("H02", "How many points did the team score?", "The team scored fewer than 60 points.", None, "comparison"),
    GoldenCase("H03", "How many residents does the town have?", "The town has about 12000 residents.", True, "comparison"),
    GoldenCase("H04", "How old is the bridge?", "The bridge is over 140 years old.", None, "comparison", note="observation: age as duration"),
    GoldenCase("H05", "How many chapters does the book have?", "The book has exactly 42 chapters.", True, "comparison", note="fixed: data-side nominal roots get synthesized predicates"),
    GoldenCase("H06", "How many tickets were sold?", "Nearly 8000 tickets were sold.", True, "comparison"),
    # ------------------------------------------------------------------ I. declarative claims (8, is_question=False)
    GoldenCase("I01", "The reservoir holds 3.2 million cubic meters of water.", "The reservoir holds 3.2 million cubic meters of water.", True, "declarative", is_question=False, note="identity control"),
    GoldenCase("I02", "The treaty was signed in 1919.", "The treaty was signed in 1918.", False, "declarative", is_question=False, note="declarative channel: NLI polarity gate flags the conflicting year"),
    GoldenCase("I03", "The expedition reached the South Pole in December 1911.", "The expedition reached the South Pole in December 1911.", True, "declarative", is_question=False),
    GoldenCase("I04", "The company employs 1200 people.", "The company reported record revenue.", None, "declarative", is_question=False, note="limitation: NUMBER vertex can over-cover weakly-typed data nouns"),
    GoldenCase("I05", "Ada Lovelace was born in London.", "Ada Lovelace was born in London in 1815.", True, "declarative", is_question=False),
    GoldenCase("I06", "The drug cures the disease.", "The drug does not cure the disease.", None, "declarative", is_question=False, note="observation: direct negation pair"),
    GoldenCase("I07", "The mayor announced the budget on Friday.", "The mayor announced the budget on Friday at noon.", True, "declarative", is_question=False),
    GoldenCase("I08", "The satellite orbits the earth.", "The telescope observes distant galaxies.", False, "declarative", is_question=False, note="unrelated doc"),
    # ------------------------------------------------------------------ J. distractors / multi-sentence docs (8)
    GoldenCase("J01", "Where was the violinist born?", "The violinist grew up in Vienna. She was born in Salzburg.", True, "distractor", note="two places; birthplace is second"),
    GoldenCase("J02", "Where was the violinist born?", "The violinist grew up in Vienna. She performed in Salzburg.", True, "distractor", note="coverage-semantics: Vienna/Salzburg are GPE targets for ?Where"),
    GoldenCase("J03", "How many goals did the striker score?", "The midfielder scored 4 goals. The striker scored 23 goals.", True, "distractor", note="fixed: containment rule picks striker over midfielder distractor"),
    GoldenCase("J04", "When did the war end?", "The war began in 1939. The war ended in 1945.", True, "distractor"),
    GoldenCase("J05", "When did the war end?", "The war began in 1939. Millions fled their homes.", True, "distractor", note="coverage-semantics: 1939 is a TEMPORAL target for ?When"),
    GoldenCase("J06", "Who painted the mural?", "The sculptor donated funds. Rivera painted the mural.", True, "distractor"),
    GoldenCase("J07", "What is the population of the city?", "The city was founded in 1608. Its population is 540000.", True, "distractor"),
    GoldenCase("J08", "Where did the ship sink?", "The ship departed from Liverpool. It sank in the North Atlantic.", True, "distractor"),
    # ------------------------------------------------------------------ K. unicode / non-ASCII (4)
    GoldenCase("K01", "Where was the treaty signed?", "The treaty was signed in Compiègne.", True, "unicode"),
    GoldenCase("K02", "When was the Oktoberfest first held?", "The Oktoberfest was first held in München in 1810.", None, "unicode", note="observation: München umlaut"),
    GoldenCase("K03", "Who wrote the book?", "José Saramago wrote the book.", True, "unicode"),
    GoldenCase("K04", "Where is the temple located?", "The temple is located in Kyōto.", True, "unicode"),
    # ------------------------------------------------------------------ L. edge cases (6)
    GoldenCase("L01", "Where was Ada Lovelace born?", "London.", False, "edge", note="coverage-semantics: bare answer 'London.' lacks subject coverage"),
    GoldenCase("L02", "How many legs does a spider have?", "A spider has 8 legs.", True, "edge"),
    GoldenCase("L03", "When was the report published?", "The report, after a long, winding, and contentious review process, was published in March.", True, "edge", note="punctuation-heavy"),
    GoldenCase("L04", "How many people live in the city?", "The city has 1,000,000 residents.", True, "edge", note="comma number"),
    GoldenCase("L05", "Who won the race?", "The race.", None, "edge", note="degenerate doc: no answer"),
    GoldenCase("L06", "When was the library built?", "The library was built in 1997 and renovated in 2015.", True, "edge", note="two years, first is the answer"),
    GoldenCase("L07", "Who painted the ceiling?", "The ceiling of the chapel was painted by Michelangelo.", True, "edge", note="fixed: data-side synthesis; passive by-phrase answer"),
    GoldenCase("L08", "How many symphonies did Beethoven write?", "Beethoven wrote nine symphonies.", None, "edge", note="observation: number as word 'nine'"),
]

assert len(CASES) == 100, f"corpus should hold exactly 100 cases, got {len(CASES)}"
