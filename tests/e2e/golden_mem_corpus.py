"""Golden memory-bank corpus for the CDR-style conflict scenario (v0.5).

Each case is a small memory bank (embodied-agent / RAG-memory style) plus the
expected set of conflicting index pairs.  Attributes are supplied explicitly
(the structured path); a separate battery exercises the unstructured NLI
fallback.

Corpus design (test-driven port of CDRs semantics):
- action/state/value/location conflict families -> expect conflicts
- similar-but-compatible banks (different object, same action; same object,
  compatible actions; updated values within tolerance) -> expect NONE
- every case is small (4-8 memories) so pairwise checks stay exhaustive
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemCase:
    id: str
    memories: tuple[tuple[str, dict[str, Any]], ...]
    expected_pairs: frozenset[tuple[int, int]]
    family: str
    note: str = ""


def _pairs(*pairs: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(sorted(p)) for p in pairs)


CASES: list[MemCase] = [
    # ------------------------------------------------------------------
    # action conflicts: same object, opposing actions
    # ------------------------------------------------------------------
    MemCase(
        "M-ACT-1",
        (
            ("Turn on the living room light", {"object": "living room light", "action": "turn on"}),
            ("Turn off the living room light", {"object": "living room light", "action": "turn off"}),
            ("Water the plants", {"object": "plants", "action": "water"}),
            ("Feed the cat", {"object": "cat", "action": "feed"}),
        ),
        _pairs((0, 1)),
        "action",
    ),
    MemCase(
        "M-ACT-2",
        (
            ("Open the window", {"object": "window", "action": "open"}),
            ("Close the window", {"object": "window", "action": "close"}),
            ("Lock the front door", {"object": "front door", "action": "lock"}),
        ),
        _pairs((0, 1)),
        "action",
    ),
    MemCase(
        "M-ACT-3",
        (
            ("Start the dishwasher", {"object": "dishwasher", "action": "start"}),
            ("Stop the dishwasher", {"object": "dishwasher", "action": "stop"}),
            ("Start the washing machine", {"object": "washing machine", "action": "start"}),
            ("Empty the trash bin", {"object": "trash bin", "action": "empty"}),
        ),
        _pairs((0, 1)),
        "action",
    ),
    MemCase(
        "M-ACT-4",
        (
            ("Charge the robot vacuum", {"object": "robot vacuum", "action": "charge"}),
            ("Deploy the robot vacuum for cleaning", {"object": "robot vacuum", "action": "deploy"}),
            ("Drain the robot vacuum battery", {"object": "robot vacuum", "action": "drain"}),
            ("Clean the kitchen floor", {"object": "kitchen floor", "action": "clean"}),
        ),
        _pairs((0, 2)),
        "action",
        note="charge vs drain are opposing actions on the same object",
    ),
    MemCase(
        "M-ACT-5",
        (
            ("Enable the security alarm", {"object": "security alarm", "action": "enable"}),
            ("Disable the security alarm", {"object": "security alarm", "action": "disable"}),
            ("Test the smoke detector", {"object": "smoke detector", "action": "test"}),
            ("Enable night mode", {"object": "night mode", "action": "enable"}),
        ),
        _pairs((0, 1)),
        "action",
    ),
    # ------------------------------------------------------------------
    # state conflicts: same object, incompatible states
    # ------------------------------------------------------------------
    MemCase(
        "M-STATE-1",
        (
            ("The front door is open", {"object": "front door", "state": "open"}),
            ("The front door is closed", {"object": "front door", "state": "closed"}),
            ("The garage door is open", {"object": "garage door", "state": "open"}),
        ),
        _pairs((0, 1)),
        "state",
    ),
    MemCase(
        "M-STATE-2",
        (
            ("The bedroom window is locked", {"object": "bedroom window", "state": "locked"}),
            ("The bedroom window is unlocked", {"object": "bedroom window", "state": "unlocked"}),
            ("The back door is locked", {"object": "back door", "state": "locked"}),
            ("The cat is asleep", {"object": "cat", "state": "asleep"}),
        ),
        _pairs((0, 1)),
        "state",
    ),
    MemCase(
        "M-STATE-3",
        (
            ("The oven is hot", {"object": "oven", "state": "hot"}),
            ("The oven is cold", {"object": "oven", "state": "cold"}),
            ("The fridge is running", {"object": "fridge", "state": "running"}),
        ),
        _pairs((0, 1)),
        "state",
    ),
    MemCase(
        "M-STATE-4",
        (
            ("The meeting room is occupied", {"object": "meeting room", "state": "occupied"}),
            ("The meeting room is vacant", {"object": "meeting room", "state": "vacant"}),
            ("The projector is on", {"object": "projector", "state": "on"}),
            ("The whiteboard is clean", {"object": "whiteboard", "state": "clean"}),
        ),
        _pairs((0, 1)),
        "state",
    ),
    MemCase(
        "M-STATE-5",
        (
            ("The water tank is full", {"object": "water tank", "state": "full"}),
            ("The water tank is empty", {"object": "water tank", "state": "empty"}),
            ("The filter needs replacement", {"object": "filter", "state": "needs replacement"}),
        ),
        _pairs((0, 1)),
        "state",
    ),
    # ------------------------------------------------------------------
    # value conflicts: same object, materially different numbers
    # ------------------------------------------------------------------
    MemCase(
        "M-VAL-1",
        (
            ("The battery level is 80 percent", {"object": "battery", "value": 80}),
            ("The battery level is 30 percent", {"object": "battery", "value": 30}),
            ("The wifi signal is strong", {"object": "wifi signal"}),
        ),
        _pairs((0, 1)),
        "value",
    ),
    MemCase(
        "M-VAL-2",
        (
            ("The thermostat reads 22 degrees", {"object": "thermostat", "value": 22}),
            ("The thermostat reads 28 degrees", {"object": "thermostat", "value": 28}),
            ("The hallway light is on", {"object": "hallway light", "state": "on"}),
        ),
        _pairs((0, 1)),
        "value",
    ),
    MemCase(
        "M-VAL-3",
        (
            ("The warehouse holds 1200 crates", {"object": "warehouse stock", "value": 1200}),
            ("The warehouse holds 950 crates", {"object": "warehouse stock", "value": 950}),
            ("The loading dock is clear", {"object": "loading dock", "state": "clear"}),
        ),
        _pairs((0, 1)),
        "value",
    ),
    MemCase(
        "M-VAL-4",
        (
            ("The robot has 4 hours of battery left", {"object": "robot battery", "value": 4}),
            ("The robot has 11 hours of battery left", {"object": "robot battery", "value": 11}),
            ("The charging dock is free", {"object": "charging dock", "state": "free"}),
        ),
        _pairs((0, 1)),
        "value",
    ),
    MemCase(
        "M-VAL-5",
        (
            ("The aquarium temperature is 24 degrees", {"object": "aquarium", "value": 24}),
            ("The aquarium temperature is 24.5 degrees", {"object": "aquarium", "value": 24.5}),
            ("Feed the fish twice a day", {"object": "fish", "action": "feed"}),
        ),
        _pairs(),
        "value",
        note="24 vs 24.5 within rel_tol=0.05 -> no conflict (tolerance boundary)",
    ),
    # ------------------------------------------------------------------
    # location conflicts: same subject, different locations
    # ------------------------------------------------------------------
    MemCase(
        "M-LOC-1",
        (
            ("The robot is in the kitchen", {"object": "robot", "state": "in kitchen"}),
            ("The robot is in the bedroom", {"object": "robot", "state": "in bedroom"}),
            ("The robot battery is charging", {"object": "robot battery", "state": "charging"}),
        ),
        _pairs((0, 1)),
        "location",
    ),
    MemCase(
        "M-LOC-2",
        (
            ("The package is at the front door", {"object": "package", "state": "at front door"}),
            ("The package is at the back door", {"object": "package", "state": "at back door"}),
            ("The mailbox is empty", {"object": "mailbox", "state": "empty"}),
        ),
        _pairs((0, 1)),
        "location",
    ),
    # ------------------------------------------------------------------
    # clean banks: similar geometry, NO conflicts (hard negatives)
    # ------------------------------------------------------------------
    MemCase(
        "M-CLEAN-1",
        (
            ("Turn on the living room light", {"object": "living room light", "action": "turn on"}),
            ("Turn on the kitchen light", {"object": "kitchen light", "action": "turn on"}),
            ("Turn off the porch light", {"object": "porch light", "action": "turn off"}),
        ),
        _pairs(),
        "clean",
        note="same action on different objects + off on a third object: no shared-object opposition",
    ),
    MemCase(
        "M-CLEAN-2",
        (
            ("The front door is closed", {"object": "front door", "state": "closed"}),
            ("Lock the front door", {"object": "front door", "action": "lock"}),
            ("The porch light is on", {"object": "porch light", "state": "on"}),
        ),
        _pairs(),
        "clean",
        note="closed door + lock action are compatible",
    ),
    MemCase(
        "M-CLEAN-3",
        (
            ("The battery level is 80 percent", {"object": "battery", "value": 80}),
            ("The battery level is 78 percent", {"object": "battery", "value": 78}),
            ("Charge the phone", {"object": "phone", "action": "charge"}),
        ),
        _pairs(),
        "clean",
        note="80 vs 78 within rel_tol -> no conflict",
    ),
    MemCase(
        "M-CLEAN-4",
        (
            ("The robot is in the kitchen", {"object": "robot", "state": "in kitchen"}),
            ("The robot is cleaning the kitchen", {"object": "robot", "action": "clean kitchen"}),
            ("The kitchen floor is wet", {"object": "kitchen floor", "state": "wet"}),
        ),
        _pairs(),
        "clean",
        note="location + ongoing action in the same place are compatible",
    ),
    MemCase(
        "M-CLEAN-5",
        (
            ("Open the window in the morning", {"object": "window", "action": "open"}),
            ("Close the window at night", {"object": "window", "action": "close"}),
            ("The window frame is wooden", {"object": "window frame", "state": "wooden"}),
        ),
        _pairs((0, 1)),
        "boundary",
        note="scheduled open/close still flags as conflict by rule semantics "
        "(temporal qualification is out of CDR scope; documents the boundary)",
    ),
    MemCase(
        "M-CLEAN-6",
        (
            ("Feed the cat at 8 AM", {"object": "cat", "action": "feed"}),
            ("Feed the cat at 6 PM", {"object": "cat", "action": "feed"}),
            ("The cat is sleeping", {"object": "cat", "state": "sleeping"}),
        ),
        _pairs(),
        "clean",
        note="same action twice a day: no opposition",
    ),
    MemCase(
        "M-CLEAN-7",
        (
            ("The garage door is open", {"object": "garage door", "state": "open"}),
            ("The garage light is on", {"object": "garage light", "state": "on"}),
            ("Park the car in the garage", {"object": "car", "action": "park"}),
        ),
        _pairs(),
        "clean",
    ),
    MemCase(
        "M-CLEAN-8",
        (
            ("The meeting starts at 10 AM", {"object": "meeting", "value": 10}),
            ("The meeting ends at 11 AM", {"object": "meeting end", "value": 11}),
            ("Book the meeting room", {"object": "meeting room", "action": "book"}),
        ),
        _pairs(),
        "clean",
        note="start/end are different objects by design",
    ),
    # ------------------------------------------------------------------
    # multi-conflict banks: several independent conflicts in one bank
    # ------------------------------------------------------------------
    MemCase(
        "M-MULTI-1",
        (
            ("Turn on the heater", {"object": "heater", "action": "turn on"}),
            ("Turn off the heater", {"object": "heater", "action": "turn off"}),
            ("The balcony door is open", {"object": "balcony door", "state": "open"}),
            ("The balcony door is closed", {"object": "balcony door", "state": "closed"}),
            ("The balcony door frame is painted", {"object": "balcony door frame", "state": "painted"}),
            ("The humidity is 60 percent", {"object": "humidity", "value": 60}),
        ),
        _pairs((0, 1), (2, 3)),
        "multi",
    ),
    MemCase(
        "M-MULTI-2",
        (
            ("The pool temperature is 26 degrees", {"object": "pool", "value": 26}),
            ("The pool temperature is 31 degrees", {"object": "pool", "value": 31}),
            ("Open the pool cover", {"object": "pool cover", "action": "open"}),
            ("Close the pool cover", {"object": "pool cover", "action": "close"}),
            ("The pool pump is running", {"object": "pool pump", "state": "running"}),
            ("Clean the pool filter weekly", {"object": "pool filter", "action": "clean"}),
        ),
        _pairs((0, 1), (2, 3)),
        "multi",
    ),
    # ------------------------------------------------------------------
    # boundary cases: near-miss attributes that must NOT fire
    # ------------------------------------------------------------------
    MemCase(
        "M-BOUND-1",
        (
            ("Turn on the light", {"object": "light", "action": "turn on"}),
            ("Turn up the light brightness", {"object": "light", "action": "turn up"}),
        ),
        _pairs(),
        "boundary",
        note="turn on vs turn up are not opposing",
    ),
    MemCase(
        "M-BOUND-2",
        (
            ("The office is warm", {"object": "office", "state": "warm"}),
            ("The office is comfortable", {"object": "office", "state": "comfortable"}),
        ),
        _pairs(),
        "boundary",
        note="warm and comfortable differ but are compatible (documents attr_conflict "
        "needing an opposites table for states; current rules fire on any "
        "difference only for action, state uses opposites list — see ruleset)",
    ),
    # ------------------------------------------------------------------
    # unstructured texts (NLI fallback path; attributes intentionally absent)
    # ------------------------------------------------------------------
    MemCase(
        "M-NLI-1",
        (
            ("The gate is open.", {}),
            ("The gate is shut.", {}),
            ("The dog is in the yard.", {}),
        ),
        _pairs((0, 1)),
        "nli-fallback",
        note="no attributes: only rico.nli_contradiction can fire",
    ),
    MemCase(
        "M-NLI-2",
        (
            ("The package arrived yesterday.", {}),
            ("The package has not arrived yet.", {}),
            ("The weather is sunny.", {}),
        ),
        _pairs((0, 1)),
        "nli-fallback",
    ),
    MemCase(
        "M-NLI-3",
        (
            ("The conference room seats twelve people.", {}),
            ("Twelve people can fit in the conference room.", {}),
            ("The projector bulb was replaced.", {}),
        ),
        _pairs(),
        "nli-fallback",
        note="paraphrase pair: NLI must not flag entailment as conflict",
    ),
]
