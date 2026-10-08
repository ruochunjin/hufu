"""Generated memory-bank corpus (~300 banks) for the CDR scenario.

Programmatic generation with expected conflict pairs known BY CONSTRUCTION:
each family instantiates slot templates (objects × actions/states/values),
adds distractor memories, and records the index pairs that the rule
semantics must flag.  Seeded RNG makes the corpus reproducible.

Families:
- action: same object, opposing actions (opposites table) -> conflict
- state: same object, opposite states -> conflict
- location: same subject, different location-valued states -> conflict
- value: same object, numeric values beyond rel_tol -> conflict
- tolerance: values WITHIN rel_tol -> no conflict
- clean-object: different objects, same action -> no conflict
- clean-compatible: same object, compatible actions/states -> no conflict
- multi: two independent conflicts + distractors in one bank
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from tests.e2e.golden_mem_corpus import MemCase

_OBJECTS = [
    "living room light", "kitchen light", "porch light", "bedroom lamp",
    "front door", "back door", "garage door", "balcony door",
    "bedroom window", "kitchen window", "bathroom window",
    "heater", "air conditioner", "ceiling fan", "dehumidifier",
    "dishwasher", "washing machine", "dryer", "oven",
    "robot vacuum", "security alarm", "smoke detector", "doorbell",
    "sprinkler system", "pool pump", "water heater", "coffee maker",
]

_ACTION_PAIRS = [
    ("turn on", "turn off"), ("open", "close"), ("start", "stop"),
    ("lock", "unlock"), ("enable", "disable"), ("charge", "drain"),
    ("fill", "empty"), ("activate", "deactivate"),
]

_COMPATIBLE_ACTIONS = [
    ("turn on", "dim"), ("open", "clean"), ("start", "monitor"),
    ("lock", "inspect"), ("enable", "configure"), ("charge", "store"),
]

_STATE_PAIRS = [
    ("open", "closed"), ("locked", "unlocked"), ("hot", "cold"),
    ("occupied", "vacant"), ("full", "empty"), ("on", "off"),
    ("asleep", "awake"), ("wet", "dry"), ("clean", "dirty"),
    ("online", "offline"), ("available", "unavailable"), ("free", "busy"),
]

_COMPATIBLE_STATES = [
    ("open", "clean"), ("locked", "painted"), ("hot", "noisy"),
    ("occupied", "bright"), ("full", "heavy"), ("on", "silent"),
]

_LOCATIONS = [
    "kitchen", "bedroom", "living room", "garage", "basement", "attic",
    "front yard", "back yard", "office", "hallway", "laundry room",
]

_SUBJECTS = ["robot", "cat", "dog", "package", "delivery drone", "cleaning bot"]

_TEMPLATES_ACTION = "{action} the {object}"
_TEMPLATES_STATE = "The {object} is {state}"
_TEMPLATES_LOC = "The {subject} is in the {place}"
_TEMPLATES_VALUE = "The {object} reading is {value} {unit}"

_VALUE_OBJECTS = [
    ("thermostat", "degrees", 18, 30),
    ("battery", "percent", 20, 95),
    ("water tank", "liters", 50, 400),
    ("air quality sensor", "ppm", 300, 1500),
    ("humidity sensor", "percent", 25, 80),
    ("tire pressure sensor", "psi", 24, 40),
]

_DISTRACTORS = [
    ("Water the plants", {"object": "plants", "action": "water"}),
    ("Feed the cat", {"object": "cat", "action": "feed"}),
    ("The mailbox is empty", {"object": "mailbox", "state": "empty"}),
    ("The wifi signal is strong", {"object": "wifi signal"}),
    ("Check the weather forecast", {"object": "weather forecast", "action": "check"}),
    ("The curtains are drawn", {"object": "curtains", "state": "drawn"}),
]


def _bank(*mems: tuple[str, dict[str, Any]]) -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple(mems)


def generate_cases(seed: int = 20261008) -> list[MemCase]:
    """Generate the full ~300-bank corpus (deterministic under ``seed``)."""

    rng = random.Random(seed)
    cases: list[MemCase] = []
    serial: dict[str, int] = {}

    def next_id(prefix: str) -> str:
        serial[prefix] = serial.get(prefix, 0) + 1
        return f"G-{prefix}-{serial[prefix]:03d}"

    def pick_distractors(n: int) -> list[tuple[str, dict[str, Any]]]:
        return rng.sample(_DISTRACTORS, n)

    # --- action conflicts (60) -------------------------------------------
    for _ in range(60):
        obj = rng.choice(_OBJECTS)
        act_a, act_b = rng.choice(_ACTION_PAIRS)
        mems = [
            (_TEMPLATES_ACTION.format(action=act_a, object=obj),
             {"object": obj, "action": act_a}),
            (_TEMPLATES_ACTION.format(action=act_b, object=obj),
             {"object": obj, "action": act_b}),
        ]
        mems += pick_distractors(rng.randint(1, 3))
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        pair = tuple(sorted((order.index(0), order.index(1))))
        cases.append(MemCase(
            next_id("ACT"), _bank(*shuffled), frozenset({pair}), "action",
        ))

    # --- state conflicts (50) --------------------------------------------
    for _ in range(50):
        obj = rng.choice(_OBJECTS)
        st_a, st_b = rng.choice(_STATE_PAIRS)
        mems = [
            (_TEMPLATES_STATE.format(object=obj, state=st_a),
             {"object": obj, "state": st_a}),
            (_TEMPLATES_STATE.format(object=obj, state=st_b),
             {"object": obj, "state": st_b}),
        ]
        mems += pick_distractors(rng.randint(1, 3))
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        pair = tuple(sorted((order.index(0), order.index(1))))
        cases.append(MemCase(
            next_id("STATE"), _bank(*shuffled), frozenset({pair}), "state",
        ))

    # --- location conflicts (30) -----------------------------------------
    for _ in range(30):
        subj = rng.choice(_SUBJECTS)
        place_a, place_b = rng.sample(_LOCATIONS, 2)
        mems = [
            (_TEMPLATES_LOC.format(subject=subj, place=place_a),
             {"object": subj, "state": f"in {place_a}"}),
            (_TEMPLATES_LOC.format(subject=subj, place=place_b),
             {"object": subj, "state": f"in {place_b}"}),
        ]
        mems += pick_distractors(rng.randint(1, 2))
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        pair = tuple(sorted((order.index(0), order.index(1))))
        cases.append(MemCase(
            next_id("LOC"), _bank(*shuffled), frozenset({pair}), "location",
        ))

    # --- value conflicts (40) --------------------------------------------
    for _ in range(40):
        obj, unit, lo, hi = rng.choice(_VALUE_OBJECTS)
        val_a = rng.randint(lo, hi)
        # at least 20% relative difference so rel_tol=0.05 always fires
        val_b = val_a + rng.choice([-1, 1]) * max(2, int(val_a * 0.2) + rng.randint(1, 5))
        val_b = max(1, val_b)
        mems = [
            (_TEMPLATES_VALUE.format(object=obj, value=val_a, unit=unit),
             {"object": obj, "value": val_a}),
            (_TEMPLATES_VALUE.format(object=obj, value=val_b, unit=unit),
             {"object": obj, "value": val_b}),
        ]
        mems += pick_distractors(rng.randint(1, 2))
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        pair = tuple(sorted((order.index(0), order.index(1))))
        cases.append(MemCase(
            next_id("VAL"), _bank(*shuffled), frozenset({pair}), "value",
        ))

    # --- tolerance boundaries (30): within rel_tol -> NO conflict --------
    big_value_objects = [spec for spec in _VALUE_OBJECTS if spec[3] >= 55]
    for _ in range(30):
        obj, unit, lo, hi = rng.choice(big_value_objects)
        val_a = rng.randint(max(lo, 50), hi)  # large base so ±2% stays within tol
        val_b = val_a + rng.choice([-1, 1]) * max(1, int(val_a * 0.02))
        mems = [
            (_TEMPLATES_VALUE.format(object=obj, value=val_a, unit=unit),
             {"object": obj, "value": val_a}),
            (_TEMPLATES_VALUE.format(object=obj, value=val_b, unit=unit),
             {"object": obj, "value": val_b}),
        ]
        mems += pick_distractors(rng.randint(1, 2))
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        cases.append(MemCase(
            next_id("TOL"), _bank(*shuffled), frozenset(), "tolerance",
            note="values within rel_tol=0.05 must not fire",
        ))

    # --- clean: different objects, same action (40) -----------------------
    for _ in range(40):
        obj_a, obj_b = rng.sample(_OBJECTS, 2)
        act = rng.choice(_ACTION_PAIRS)[0]
        mems = [
            (_TEMPLATES_ACTION.format(action=act, object=obj_a),
             {"object": obj_a, "action": act}),
            (_TEMPLATES_ACTION.format(action=act, object=obj_b),
             {"object": obj_b, "action": act}),
        ]
        mems += pick_distractors(rng.randint(1, 2))
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        cases.append(MemCase(
            next_id("COBJ"), _bank(*shuffled), frozenset(), "clean-object",
        ))

    # --- clean: same object, compatible actions/states (40) ---------------
    for i in range(40):
        obj = rng.choice(_OBJECTS)
        if i % 2 == 0:
            act_a, act_b = rng.choice(_COMPATIBLE_ACTIONS)
            mems = [
                (_TEMPLATES_ACTION.format(action=act_a, object=obj),
                 {"object": obj, "action": act_a}),
                (_TEMPLATES_ACTION.format(action=act_b, object=obj),
                 {"object": obj, "action": act_b}),
            ]
        else:
            st_a, st_b = rng.choice(_COMPATIBLE_STATES)
            mems = [
                (_TEMPLATES_STATE.format(object=obj, state=st_a),
                 {"object": obj, "state": st_a}),
                (_TEMPLATES_STATE.format(object=obj, state=st_b),
                 {"object": obj, "state": st_b}),
            ]
        mems += pick_distractors(rng.randint(1, 2))
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        cases.append(MemCase(
            next_id("CCOMP"), _bank(*shuffled), frozenset(), "clean-compatible",
        ))

    # --- multi-conflict banks (30) ----------------------------------------
    for _ in range(30):
        obj_a, obj_b = rng.sample(_OBJECTS, 2)
        act_a, act_b = rng.choice(_ACTION_PAIRS)
        st_a, st_b = rng.choice(_STATE_PAIRS)
        mems = [
            (_TEMPLATES_ACTION.format(action=act_a, object=obj_a),
             {"object": obj_a, "action": act_a}),
            (_TEMPLATES_ACTION.format(action=act_b, object=obj_a),
             {"object": obj_a, "action": act_b}),
            (_TEMPLATES_STATE.format(object=obj_b, state=st_a),
             {"object": obj_b, "state": st_a}),
            (_TEMPLATES_STATE.format(object=obj_b, state=st_b),
             {"object": obj_b, "state": st_b}),
        ]
        mems += pick_distractors(1)
        order = list(range(len(mems)))
        rng.shuffle(order)
        shuffled = [mems[i] for i in order]
        pair1 = tuple(sorted((order.index(0), order.index(1))))
        pair2 = tuple(sorted((order.index(2), order.index(3))))
        cases.append(MemCase(
            next_id("MULTI"), _bank(*shuffled), frozenset({pair1, pair2}), "multi",
        ))

    return cases


CASES: list[MemCase] = generate_cases()
