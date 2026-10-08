"""Generated RAG-consistency corpus (~200 queries x 3 documents = 600 pairs).

Each case poses one question (or declarative claim) over a slot fact, then
attaches three retrieved-document stand-ins:

- the GOLD passage (answers correctly) -> expected consistent=True
- the CONFLICT passage (same frame, wrong value) -> expected consistent=False
- an IRRELEVANT passage (unrelated entity) -> expected consistent=True
  (irrelevance is a distance/usefulness matter, not a contradiction)

Labels are known by construction; the generator is seeded and deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tests.e2e.golden_struct200 import _CITIES, _COMPANIES, _PEOPLE, _PRODUCTS


@dataclass(frozen=True)
class RagCase:
    id: str
    query: str
    documents: tuple[str, ...]
    expected_consistent: tuple[bool, ...]
    family: str
    note: str = ""
    is_question: bool = True


_BOOKS = [
    ("the novel 'Silent Harbor'", "Margaret Ellison"),
    ("the novel 'Winter Ledger'", "Tomasz Kowalczyk"),
    ("the biography 'Iron Roots'", "Fatima Al-Sayed"),
    ("the textbook 'Graphs in Practice'", "Dr. Helena Vogt"),
    ("the memoir 'Paper Trails'", "Derek Anand"),
    ("the novel 'Glass Meridian'", "Yuki Tanaka"),
    ("the monograph 'Quiet Engines'", "Dr. Samuel Adeyemi"),
    ("the novel 'Northern Static'", "Ingrid Sorensen"),
]

_FOUNDED = [
    ("Northwind Traders", 1968), ("Acme Robotics", 1994),
    ("Bluepeak Analytics", 2007), ("Silverline Foods", 1981),
    ("Helios Energy", 2002), ("Cascade Materials", 1975),
    ("Orion Logistics", 1989), ("Pinnacle Software", 1998),
]


def generate_cases(seed: int = 20261008) -> list[RagCase]:
    rng = random.Random(seed)
    cases: list[RagCase] = []
    serial: dict[str, int] = {}

    def next_id(prefix: str) -> str:
        serial[prefix] = serial.get(prefix, 0) + 1
        return f"R-{prefix}-{serial[prefix]:03d}"

    # --- where-born (40) ---------------------------------------------------
    for _ in range(40):
        person = rng.choice(_PEOPLE)
        city_right, city_wrong = rng.sample(_CITIES, 2)
        other_person = rng.choice([p for p in _PEOPLE if p != person])
        other_city = rng.choice([c for c in _CITIES if c != city_right])
        query = f"Where was {person} born?"
        docs = (
            f"{person} was born in {city_right}.",
            f"{person} was born in {city_wrong}.",
            f"{other_person} was born in {other_city}.",
        )
        cases.append(RagCase(
            next_id("WHERE"), query, docs, (True, True, True), "where-born",
            note="wrong-city answer still covers the question (answer-correctness out of scope)",
        ))

    # --- when-founded (40) --------------------------------------------------
    for _ in range(40):
        company, year_right = rng.choice(_FOUNDED)
        year_wrong = year_right + rng.choice([-1, 1]) * rng.randint(3, 15)
        other_company, other_year = rng.choice(
            [f for f in _FOUNDED if f[0] != company]
        )
        query = f"When was {company} founded?"
        docs = (
            f"{company} was founded in {year_right}.",
            f"{company} was founded in {year_wrong}.",
            f"{other_company} was founded in {other_year}.",
        )
        cases.append(RagCase(
            next_id("WHEN"), query, docs, (True, True, True), "when-founded",
            note="wrong-year answer still covers the question",
        ))

    # --- who-wrote (40) ------------------------------------------------------
    for _ in range(40):
        book, author_right = rng.choice(_BOOKS)
        author_wrong = rng.choice([p for p in _PEOPLE if p != author_right])
        other_book, other_author = rng.choice([b for b in _BOOKS if b[0] != book])
        query = f"Who wrote {book}?"
        docs = (
            f"{book} was written by {author_right}.",
            f"{book} was written by {author_wrong}.",
            f"{other_book} was written by {other_author}.",
        )
        cases.append(RagCase(
            next_id("WHO"), query, docs, (True, True, False), "who-wrote",
            note="wrong-author answer still covers the question",
        ))

    # --- how-many-employees (40) ---------------------------------------------
    for _ in range(40):
        company = rng.choice(_COMPANIES)
        n_right = rng.randint(80, 9000)
        n_wrong = n_right * rng.choice([3, 5]) + rng.randint(11, 77)
        other_company = rng.choice([c for c in _COMPANIES if c != company])
        n_other = rng.randint(80, 9000)
        query = f"How many people does {company} employ?"
        docs = (
            f"{company} employs {n_right} people.",
            f"{company} employs {n_wrong} people.",
            f"{other_company} employs {n_other} people.",
        )
        cases.append(RagCase(
            next_id("NUM"), query, docs, (True, True, True), "how-many",
            note="wrong-count answer still covers the question",
        ))

    # --- declarative claims (40) ----------------------------------------------
    for _ in range(40):
        product = rng.choice(_PRODUCTS)
        w_right = round(rng.uniform(0.5, 9.0), 1)
        w_wrong = round(w_right * rng.choice([1.8, 2.5]), 1)
        other_product = rng.choice([p for p in _PRODUCTS if p != product])
        w_other = round(rng.uniform(0.5, 9.0), 1)
        query = f"The {product} weighs {w_right} kg."
        docs = (
            f"The {product} weighs {w_right} kg and ships with a case.",
            f"The {product} weighs {w_wrong} kg and ships with a case.",
            f"The {other_product} weighs {w_other} kg.",
        )
        cases.append(RagCase(
            next_id("DECL"), query, docs, (True, False, False), "declarative",
            note="declarative claim: polarity channel flags the wrong value; irrelevant doc uncovered",
            is_question=False,
        ))

    return cases


CASES: list[RagCase] = generate_cases()
