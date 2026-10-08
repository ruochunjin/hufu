"""Generated structured-document corpus (~200 docs) for the strict star engine.

Fact-sheet / record-style documents — the scope where the strict profile is
meant to excel.  Every document is 2-3 short declarative sentences generated
from slot templates; the label (contradiction or not) is known by
construction.

Families:
- bio-loc / bio-year: person facts with swapped location/year
- spec-num: product/org numeric specs with swapped values
- org-loc / org-num: headquarters/employee-count swaps
- role: asymmetric-verb subject/object reversal
- antonym: antonymous predicates on one subject
- negation: polarity flip on an identical frame
- clean-*: same skeleton, consistent or complementary facts
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class StructCase:
    id: str
    text: str
    expected: bool
    family: str
    note: str = ""


_PEOPLE = [
    "Dr. Elena Marsh", "Prof. David Chen", "Sarah Okafor", "Michael Novak",
    "Dr. Priya Raman", "James Whitfield", "Laura Bianchi", "Tomás Herrera",
    "Dr. Ingrid Sorensen", "Kevin Osei", "Rachel Kim", "Dr. Omar Farouk",
]
_CITIES = [
    "Portland", "Adelaide", "Lyon", "Osaka", "Tampere", "Curitiba",
    "Graz", "Halifax", "Bergen", "Valencia", "Leeds", "Nagoya",
]
_COMPANIES = [
    "Northwind Traders", "Acme Robotics", "Bluepeak Analytics",
    "Silverline Foods", "Helios Energy", "Cascade Materials",
    "Orion Logistics", "Pinnacle Software", "Redwood Pharma",
    "Summit Aerospace", "Vertex Instruments", "Harbor Freight Systems",
]
_PRODUCTS = [
    "Aurora X1 drone", "Titan PRO battery pack", "Nimbus weather station",
    "Voyager GPS watch", "Forge 3D printer", "Atlas cargo e-bike",
    "Pulse Pro heart-rate monitor", "Echo smart speaker", "Vertex Pro laser rangefinder",
    "Summit camping stove",
]
_OBJECTS = [
    "safety test", "quality inspection", "emission test", "stress test",
    "calibration check", "pressure test",
]
# All ten are genuine subject/object REVERSALS on asymmetric verbs (R-ROLE);
# a same-frame subject swap ("A acquired X. B acquired X.") is an R-VALUE
# case, not a reversal — do not mix the two here.
_EVENTS = [
    ("Northwind Traders acquired Silverline Foods", "Silverline Foods acquired Northwind Traders"),
    ("The laboratory hired Dr. Elena Marsh", "Dr. Elena Marsh hired the laboratory"),
    ("The startup poached the senior engineer", "The senior engineer poached the startup"),
    ("The museum sold the painting to the collector", "The collector sold the painting to the museum"),
    ("The agency appointed Dr. Reyes as director", "Dr. Reyes appointed the agency as director"),
    ("The studio fired the lead director", "The lead director fired the studio"),
    ("The university promoted Prof. David Chen", "Prof. David Chen promoted the university"),
    ("The landlord evicted the tenant", "The tenant evicted the landlord"),
    ("The officer arrested the suspect", "The suspect arrested the officer"),
    ("The club expelled the member", "The member expelled the club"),
]


def generate_cases(seed: int = 20261008) -> list[StructCase]:
    rng = random.Random(seed)
    cases: list[StructCase] = []
    serial: dict[str, int] = {}

    def next_id(prefix: str) -> str:
        serial[prefix] = serial.get(prefix, 0) + 1
        return f"D-{prefix}-{serial[prefix]:03d}"

    # --- bio location conflicts (20) -------------------------------------
    for _ in range(20):
        person = rng.choice(_PEOPLE)
        city_a, city_b = rng.sample(_CITIES, 2)
        cases.append(StructCase(
            next_id("BIOL"),
            f"{person} was born in {city_a}. {person} was born in {city_b}.",
            True, "bio-loc",
        ))

    # --- bio year conflicts (15) ------------------------------------------
    for _ in range(15):
        person = rng.choice(_PEOPLE)
        year_a = rng.randint(1955, 1995)
        year_b = year_a + rng.choice([-1, 1]) * rng.randint(1, 8)
        cases.append(StructCase(
            next_id("BIOY"),
            f"{person} was born in {year_a}. {person} was born in {year_b}.",
            True, "bio-year",
        ))

    # --- product numeric spec conflicts (20) ------------------------------
    for _ in range(20):
        product = rng.choice(_PRODUCTS)
        kind = rng.choice(["weighs", "costs", "lasts"])
        if kind == "weighs":
            va = round(rng.uniform(0.5, 9.0), 1)
            vb = round(va * rng.choice([1.5, 2.0, 0.5]), 1)
            text = f"The {product} weighs {va} kg. The {product} weighs {vb} kg."
        elif kind == "costs":
            va = rng.randint(50, 900)
            vb = va * rng.choice([2, 3]) + rng.randint(5, 50)
            text = f"The {product} costs {va} dollars. The {product} costs {vb} dollars."
        else:
            va = rng.randint(4, 40)
            vb = va + rng.choice([-1, 1]) * rng.randint(3, 12)
            text = (
                f"The {product} lasts {va} hours on a charge. "
                f"The {product} lasts {vb} hours on a charge."
            )
        cases.append(StructCase(next_id("SPEC"), text, True, "spec-num"))

    # --- org location conflicts (15) --------------------------------------
    for _ in range(15):
        company = rng.choice(_COMPANIES)
        city_a, city_b = rng.sample(_CITIES, 2)
        cases.append(StructCase(
            next_id("ORGL"),
            f"{company} is headquartered in {city_a}. {company} is headquartered in {city_b}.",
            True, "org-loc",
        ))

    # --- org employee-count conflicts (10) --------------------------------
    for _ in range(10):
        company = rng.choice(_COMPANIES)
        na = rng.randint(50, 9000)
        nb = na * rng.choice([2, 3, 4]) + rng.randint(10, 99)
        cases.append(StructCase(
            next_id("ORGN"),
            f"{company} employs {na} people. {company} employs {nb} people.",
            True, "org-num",
        ))

    # --- role reversals (10) ----------------------------------------------
    for text_a, text_b in _EVENTS:
        cases.append(StructCase(
            next_id("ROLE"), f"{text_a}. {text_b}.", True, "role",
        ))

    # --- antonym conflicts (10) -------------------------------------------
    _ant = [
        ("The stock price rose sharply at the open", "The stock price fell sharply at the open"),
        ("The prototype passed the safety test", "The prototype failed the safety test"),
        ("The startup expanded its workforce", "The startup shrank its workforce"),
        ("The device improved the signal", "The device worsened the signal"),
        ("The committee approved the proposal", "The committee rejected the proposal"),
        ("The team supported the initiative", "The team opposed the initiative"),
        ("The vaccine survived the trial", "The vaccine died in the trial"),
        ("The alloy strengthened the frame", "The alloy weakened the frame"),
        ("The reform improved the system", "The reform worsened the system"),
        ("The army advanced toward the border", "The army retreated toward the border"),
    ]
    for text_a, text_b in _ant:
        cases.append(StructCase(
            next_id("ANT"), f"{text_a}. {text_b}.", True, "antonym",
        ))

    # --- negation conflicts (10) -------------------------------------------
    for obj in _OBJECTS[:10]:
        cases.append(StructCase(
            next_id("NEG"),
            f"The prototype passed the {obj}. The prototype did not pass the {obj}.",
            True, "negation",
        ))

    # --- clean: consistent restatements (40) -------------------------------
    for i in range(40):
        variant = i % 4
        person = rng.choice(_PEOPLE)
        company = rng.choice(_COMPANIES)
        product = rng.choice(_PRODUCTS)
        city = rng.choice(_CITIES)
        if variant == 0:
            text = (
                f"{person} was born in {city}. {person} grew up in {city}."
            )
        elif variant == 1:
            year = rng.randint(1955, 1995)
            text = (
                f"{person} was born in {year}. "
                f"{person} received a doctorate in {year + rng.randint(22, 30)}."
            )
        elif variant == 2:
            na = rng.randint(50, 9000)
            text = (
                f"{company} employs {na} people. "
                f"{company} is headquartered in {rng.choice(_CITIES)}."
            )
        else:
            va = round(rng.uniform(0.5, 9.0), 1)
            text = (
                f"The {product} weighs {va} kg. "
                f"The {product} ships with a carrying case."
            )
        cases.append(StructCase(next_id("CLEAN"), text, False, "clean"))

    # --- clean: different entities, parallel structure (20) ---------------
    for _ in range(20):
        person_a, person_b = rng.sample(_PEOPLE, 2)
        city_a, city_b = rng.sample(_CITIES, 2)
        cases.append(StructCase(
            next_id("CDIFF"),
            f"{person_a} was born in {city_a}. {person_b} was born in {city_b}.",
            False, "clean-diff",
            note="different subjects: parallel facts, no conflict",
        ))

    # --- clean: multi-fact records (20) ------------------------------------
    for _ in range(20):
        company = rng.choice(_COMPANIES)
        city = rng.choice(_CITIES)
        na = rng.randint(50, 9000)
        year = rng.randint(1975, 2015)
        cases.append(StructCase(
            next_id("CREC"),
            f"{company} is headquartered in {city}. "
            f"{company} employs {na} people. "
            f"{company} was founded in {year}.",
            False, "clean-record",
        ))

    return cases


CASES: list[StructCase] = generate_cases()
