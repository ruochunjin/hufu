"""Golden document corpus v300: 300 documents across conflict types and scenarios.

Composition (by design, documented in the working record):
- 50 hand-crafted minimal rule-battery docs (from the v100 corpus);
- 100 ContraDoc positive docs, stratified proportionally over contra_type
  (Content/Perspective/Negation/Emotion/Numeric/Factual/Causal/Relation);
- 100 ContraDoc negative docs, stratified proportionally over doc_type
  (story/news/wiki);
- 50 synthetic hard-negative narrative docs (biography, history, news, and
  technical-report templates with internally consistent facts — the shapes
  that previously produced false flags).
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e.golden_doc_corpus import HAND_CASES, DocCase

_DATA_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "ContraDoc.json"


def _stratified(items: list, key_fn, total: int) -> list:
    """Pick `total` items proportionally across key groups (topped up)."""

    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    picked: list = []
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    quota = {k: max(1, round(total * len(groups[k]) / len(items))) for k in keys}
    for key in keys:
        pool = groups[key]
        step = max(1, len(pool) // quota[key])
        picked.extend(pool[::step][: quota[key]])
    if len(picked) < total:
        chosen = set(map(id, picked))
        for item in items:
            if len(picked) >= total:
                break
            if id(item) not in chosen:
                picked.append(item)
    return picked[:total]


def _load_contradoc() -> dict:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def contradoc_pos_samples(total: int = 100) -> list[DocCase]:
    data = _load_contradoc()
    docs = list(data["pos"].items())
    picked = _stratified(
        docs,
        lambda kv: (kv[1].get("contra_type") or ["Other"])[0],
        total,
    )
    return [
        DocCase(
            id=f"CD-POS-{doc_id}",
            text=doc["text"],
            expected=True,
            family="contradoc-pos",
            note=str(doc.get("contra_type", "")),
        )
        for doc_id, doc in picked
    ]


def contradoc_neg_samples(total: int = 100) -> list[DocCase]:
    data = _load_contradoc()
    docs = list(data["neg"].items())
    picked = _stratified(docs, lambda kv: kv[1].get("doc_type", "?"), total)
    return [
        DocCase(
            id=f"CD-NEG-{doc_id}",
            text=doc["text"],
            expected=False,
            family="contradoc-neg",
            note=str(doc.get("doc_type", "")),
        )
        for doc_id, doc in picked
    ]


# --- Synthetic hard negatives (consistent narratives in the FP-prone shapes)

_BIO = (
    "{name} was born in {city_a} in {year}. {pronoun} grew up in {city_b} and "
    "studied music at the conservatory. {pronoun} later moved to {city_c}, "
    "where {pronoun_lower} taught for nearly two decades. Critics praised "
    "{name}'s discipline and warmth in the classroom."
)
_HIST = (
    "The expedition left {city_a} in the spring of {year}. After weeks at sea "
    "the crew reached {place_a} and wintered there. In the following spring "
    "they pushed on to {place_b}, losing two sled dogs but no men. The leader "
    "later wrote that the hardest part had been the waiting."
)
_NEWS = (
    "The company announced on Tuesday that revenue rose {pct1} percent to "
    "${amount} million in the third quarter, driven by strong demand in its "
    "cloud division. The company said it now employs {staff} people across "
    "offices in {city_a} and {city_b}. Analysts had expected slower growth."
)
_TECH = (
    "Version {ver} of the library was released in {month} {year}. The release "
    "notes list {count} fixed issues, including a race condition in the "
    "scheduler and a memory leak in the parser. The maintainers recommend "
    "upgrading for all production deployments."
)

_PEOPLE = [
    ("Clara Wainwright", "She", "she", "York", "London", "Edinburgh", "1902"),
    ("Elias Thorn", "He", "he", "Oslo", "Bergen", "Trondheim", "1898"),
    ("Marguerite Voss", "She", "she", "Lyon", "Paris", "Brussels", "1911"),
    ("Nikolai Petrov", "He", "he", "Kazan", "Moscow", "Tbilisi", "1895"),
    ("Ingrid Halvorsen", "She", "she", "Aarhus", "Copenhagen", "Malmö", "1908"),
    ("Tomás Herrera", "He", "he", "Sevilla", "Madrid", "Lisbon", "1904"),
    ("Beatrice Kwan", "She", "she", "Penang", "Singapore", "Jakarta", "1913"),
    ("Aldo Ferrante", "He", "he", "Turin", "Milan", "Zurich", "1899"),
    ("Sofia Lindqvist", "She", "she", "Uppsala", "Stockholm", "Helsinki", "1906"),
    ("Ravi Menon", "He", "he", "Kochi", "Mumbai", "Dubai", "1901"),
]
_EXPEDITIONS = [
    ("Plymouth", "1910", "the Falkland Islands", "the Ross Sea"),
    ("Hobart", "1912", "Macquarie Island", "the Adélie Coast"),
    ("Tromsø", "1908", "Svalbard", "Franz Josef Land"),
    ("Brest", "1915", "the Azores", "the Cape Verde islands"),
    ("Karlskrona", "1906", "Iceland", "the Greenland coast"),
]
_COMPANIES = [
    ("8", "420", "1200", "Austin", "Denver", "2.4", "March", "2023", "37"),
    ("12", "310", "860", "Tallinn", "Riga", "1.9", "June", "2024", "52"),
    ("5", "980", "3400", "Pune", "Hyderabad", "3.1", "October", "2022", "41"),
    ("9", "265", "640", "Cork", "Galway", "2.8", "April", "2024", "28"),
    ("14", "150", "2100", "Da Nang", "Hanoi", "2.2", "July", "2023", "63"),
]


def synthetic_negatives() -> list[DocCase]:
    cases: list[DocCase] = []
    for index, (name, pron, pron_l, city_a, city_b, city_c, year) in enumerate(_PEOPLE):
        cases.append(
            DocCase(
                id=f"SYN-BIO-{index + 1}",
                text=_BIO.format(
                    name=name, pronoun=pron, pronoun_lower=pron_l,
                    city_a=city_a, city_b=city_b, city_c=city_c, year=year,
                ),
                expected=False,
                family="synthetic-clean",
                note="biography, consistent places/dates",
            )
        )
    for index, (city_a, year, place_a, place_b) in enumerate(_EXPEDITIONS):
        cases.append(
            DocCase(
                id=f"SYN-HIST-{index + 1}",
                text=_HIST.format(city_a=city_a, year=year, place_a=place_a, place_b=place_b),
                expected=False,
                family="synthetic-clean",
                note="expedition narrative, consistent timeline",
            )
        )
    for index, (pct1, amount, staff, city_a, city_b, ver, month, year, count) in enumerate(_COMPANIES):
        cases.append(
            DocCase(
                id=f"SYN-NEWS-{index + 1}",
                text=_NEWS.format(pct1=pct1, amount=amount, staff=staff, city_a=city_a, city_b=city_b),
                expected=False,
                family="synthetic-clean",
                note="earnings report, consistent numbers",
            )
        )
        cases.append(
            DocCase(
                id=f"SYN-TECH-{index + 1}",
                text=_TECH.format(ver=ver, month=month, year=year, count=count),
                expected=False,
                family="synthetic-clean",
                note="release notes, consistent metadata",
            )
        )
    # 30 more: duplicate with role/tense variations to reach 50
    for index, (name, pron, pron_l, city_a, city_b, city_c, year) in enumerate(_PEOPLE):
        cases.append(
            DocCase(
                id=f"SYN-BIOV-{index + 1}",
                text=(
                    f"{name} taught in {city_c} for almost twenty years. Earlier, "
                    f"{pron.lower()} had studied music at the conservatory in {city_b}. "
                    f"{name} was born in {city_a} in {year}, and colleagues remember "
                    f"{pron_l} warmly to this day."
                ),
                expected=False,
                family="synthetic-clean",
                note="biography reordered, consistent facts",
            )
        )
    for index, (city_a, year, place_a, place_b) in enumerate(_EXPEDITIONS):
        cases.append(
            DocCase(
                id=f"SYN-HISTV-{index + 1}",
                text=(
                    f"Two sled dogs were lost crossing to {place_b}, but no men. "
                    f"The crew had wintered at {place_a} after sailing from {city_a} "
                    f"in the spring of {year}. The leader later wrote that the "
                    "waiting had been the hardest part."
                ),
                expected=False,
                family="synthetic-clean",
                note="expedition reordered, consistent timeline",
            )
        )
    for index, (pct1, amount, staff, city_a, city_b, ver, month, year, count) in enumerate(_COMPANIES[:10]):
        cases.append(
            DocCase(
                id=f"SYN-MIX-{index + 1}",
                text=(
                    f"Analysts had expected slower growth, but {pct1} percent revenue "
                    f"growth to ${amount} million beat expectations, the company said "
                    f"Tuesday. It now employs {staff} people in {city_a} and {city_b}, "
                    f"and credits strong cloud demand. The report also notes version "
                    f"{ver} shipped in {month} {year} with {count} fixed issues."
                ),
                expected=False,
                family="synthetic-clean",
                note="mixed news/tech, consistent numbers",
            )
        )
    for index, (pct1, amount, staff, city_a, city_b, ver, month, year, count) in enumerate(_COMPANIES):
        cases.append(
            DocCase(
                id=f"SYN-MIXV-{index + 1}",
                text=(
                    f"Version {ver} shipped in {month} {year} with {count} fixed issues, "
                    f"including a race condition in the scheduler. The same week the "
                    f"company reported {pct1} percent revenue growth to ${amount} million "
                    f"and said it now employs {staff} people in {city_a} and {city_b}."
                ),
                expected=False,
                family="synthetic-clean",
                note="mixed tech/news reordered, consistent numbers",
            )
        )
    return cases


CASES: list[DocCase] = (
    list(HAND_CASES) + contradoc_pos_samples(100) + contradoc_neg_samples(100) + synthetic_negatives()
)
