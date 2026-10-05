"""Golden document corpus for the star-graph contradiction engine (~120 docs).

Two tiers:
- Hand-crafted minimal documents targeting each rule (S-* ids): numeric,
  date, place, person, negation, antonym, role-reversal, temporal-recurring
  guards, and clean controls of the same shapes.
- Real ContraDoc samples (CD-* ids): 25 contradictory (pos) + 25 clean
  (neg) documents, validating the engine on narrative text with entity-rich
  star structure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocCase:
    id: str
    text: str
    expected: bool  # True = contradictory, False = clean
    family: str
    note: str = ""


HAND_CASES: list[DocCase] = [
    # ---- S-NUM: numeric conflicts (expect contradiction)
    DocCase("S-NUM-1", "The regiment lost 300 men at the ridge. The regiment lost 3000 men at the ridge.", True, "numeric"),
    DocCase("S-NUM-2", "The company employs 1200 people. The company employs 2100 people.", True, "numeric"),
    DocCase("S-NUM-3", "The reservoir holds 3.2 million cubic meters of water. The reservoir holds 3.3 million cubic meters of water.", True, "numeric"),
    DocCase("S-NUM-4", "The striker scored 23 goals this season. The striker scored 32 goals this season.", True, "numeric"),
    DocCase("S-NUM-5", "The stadium seats 65000 fans. The stadium seats 56000 fans.", True, "numeric"),
    # ---- S-DATE: date conflicts
    DocCase("S-DATE-1", "The treaty was signed in 1918. The treaty was signed in 1919.", True, "date"),
    DocCase("S-DATE-2", "The company went public in 2019. The company went public in 2021.", True, "date"),
    DocCase("S-DATE-3", "The bridge was opened in May 1883. The bridge was opened in June 1883.", True, "date"),
    DocCase("S-DATE-4", "The reactor came online in 1983. The reactor came online in 1985.", True, "date"),
    DocCase("S-DATE-5", "The library was built in 1997 and renovated in 2015. The library was built in 2015 and renovated in 1997.", True, "date"),
    # ---- S-PLACE: place conflicts
    DocCase("S-PLACE-1", "Ernest Hemingway was born in Oak Park. Ernest Hemingway was born in Springfield.", True, "place"),
    DocCase("S-PLACE-2", "The headquarters is in Austin, Texas. The headquarters is in Boston, Massachusetts.", True, "place"),
    DocCase("S-PLACE-3", "The cathedral was built in Chartres. The cathedral was built in Reims.", True, "place"),
    DocCase("S-PLACE-4", "The probe landed on Mars in 2021. The probe landed on Venus in 2021.", True, "place"),
    DocCase("S-PLACE-5", "The ship sank in the North Atlantic. The ship sank in the Mediterranean.", True, "place"),
    # ---- S-PERSON: person conflicts
    DocCase("S-PERSON-1", "Jane Austen wrote the novel Emma. Charlotte Brontë wrote the novel Emma.", True, "person"),
    DocCase("S-PERSON-2", "The CEO of the company is Mary Chen. The CEO of the company is John Smith.", True, "person"),
    DocCase("S-PERSON-3", "Alexander Fleming discovered penicillin. Howard Florey discovered penicillin.", True, "person"),
    DocCase("S-PERSON-4", "The committee appointed Dr. Reyes as director. The committee appointed Dr. Vance as director.", True, "person"),
    DocCase("S-PERSON-5", "Professor Ito taught the course. Professor Yamada taught the course.", True, "person"),
    # ---- S-NEG: negation conflicts
    DocCase("S-NEG-1", "Yorick stayed in Paris for a few days. Yorick never visited Paris on his journey.", True, "negation"),
    DocCase("S-NEG-2", "The queen attended the ceremony in person. The queen did not attend the ceremony.", True, "negation"),
    DocCase("S-NEG-3", "The drug is approved for children. The drug is not approved for children.", True, "negation"),
    DocCase("S-NEG-4", "The miners struck gold in the second shaft. The miners found nothing in the second shaft.", True, "negation"),
    DocCase("S-NEG-5", "The river froze solid that winter. The river never froze that winter.", True, "negation"),
    # ---- S-ANT: antonym conflicts
    DocCase("S-ANT-1", "The museum is open on Mondays. The museum is closed on Mondays.", True, "antonym"),
    DocCase("S-ANT-2", "The stock price rose sharply. The stock price fell sharply.", True, "antonym"),
    DocCase("S-ANT-3", "The project was a success. The project was a failure.", True, "antonym"),
    DocCase("S-ANT-4", "The machine increases the pressure. The machine decreases the pressure.", True, "antonym"),
    DocCase("S-ANT-5", "The senator supported the bill. The senator opposed the bill.", True, "antonym"),
    # ---- S-ROLE: role reversals
    DocCase("S-ROLE-1", "The collector sold the painting to the museum. The museum sold the painting to the collector.", True, "role"),
    DocCase("S-ROLE-2", "The startup hired the engineer. The engineer hired the startup.", True, "role"),
    DocCase("S-ROLE-3", "The dog bit the mailman. The mailman bit the dog.", True, "role"),
    DocCase("S-ROLE-4", "The landlord evicted the tenant. The tenant evicted the landlord.", True, "role"),
    DocCase("S-ROLE-5", "The company acquired the startup. The startup acquired the company.", True, "role"),
    # ---- S-CLEAN: consistent controls
    DocCase("S-CLEAN-1", "Ernest Hemingway was born in Oak Park. He wrote many novels.", False, "clean"),
    DocCase("S-CLEAN-2", "The violinist grew up in Vienna. She performed in Salzburg for years.", False, "clean"),
    DocCase("S-CLEAN-3", "The collector sold the painting to the museum. The painting was sold to the museum by the collector.", False, "clean", note="passive alternation"),
    DocCase("S-CLEAN-4", "Three men went into the cave. Two men came out at dawn.", False, "clean", note="sequential numbers, not a conflict"),
    DocCase("S-CLEAN-5", "The museum is open on Mondays. The museum is open on Tuesdays.", False, "clean", note="recurring temporal is inclusive"),
    DocCase("S-CLEAN-6", "The treaty was signed in 1918. It ended the war in Europe.", False, "clean"),
    DocCase("S-CLEAN-7", "The company employs 1200 people. It is hiring aggressively.", False, "clean"),
    DocCase("S-CLEAN-8", "Ada Lovelace was born in London in 1815. She was an English mathematician.", False, "clean"),
    DocCase("S-CLEAN-9", "The striker scored 23 goals this season. The midfielder scored 4 goals.", False, "clean", note="different players, different numbers"),
    DocCase("S-CLEAN-10", "The marathon race is 42.195 kilometers. The marathon race is 42195 meters.", False, "clean", note="unit conversion equivalent"),
    DocCase("S-CLEAN-11", "The cathedral is 800 years old. The cathedral dominates the old town.", False, "clean"),
    DocCase("S-CLEAN-12", "The committee appointed Dr. Reyes. The committee announced the new budget.", False, "clean"),
    DocCase("S-CLEAN-13", "The reservoir holds 3.2 million cubic meters of water. It was filled after the 1998 dam expansion.", False, "clean"),
    DocCase("S-CLEAN-14", "The violinist was born in Salzburg. She grew up in Vienna and debuted at fifteen.", False, "clean", note="different life events, different places"),
    DocCase("S-CLEAN-15", "The flight departed two hours late. The flight arrived on schedule.", True, "clean", note="late departure + on-schedule arrival is a real tension — expected contradiction via NLI; documents the boundary"),
]


def contra_doc_samples(n_pos: int = 25, n_neg: int = 25) -> list[DocCase]:
    """Sample real ContraDoc documents (loaded lazily from benchmarks data)."""

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "ContraDoc.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[DocCase] = []
    for index, (doc_id, doc) in enumerate(data["pos"].items()):
        if index >= n_pos:
            break
        cases.append(
            DocCase(
                id=f"CD-POS-{doc_id}",
                text=doc["text"],
                expected=True,
                family="contradoc-pos",
                note=str(doc.get("contra_type", "")),
            )
        )
    for index, (doc_id, doc) in enumerate(data["neg"].items()):
        if index >= n_neg:
            break
        cases.append(
            DocCase(
                id=f"CD-NEG-{doc_id}",
                text=doc["text"],
                expected=False,
                family="contradoc-neg",
                note=str(doc.get("doc_type", "")),
            )
        )
    return cases


CASES: list[DocCase] = HAND_CASES + contra_doc_samples()
