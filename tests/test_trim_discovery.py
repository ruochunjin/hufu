"""Dependency-free tests for evidence trimming and rule discovery."""

from __future__ import annotations

import json
import unittest

from hufu.rules import Item, RuleEngine
from hufu.rules.discovery import discover_rules
from hufu.scenarios.trim import trim_evidence


class TrimEvidenceTests(unittest.TestCase):
    def test_keep_recent_drops_older_side(self) -> None:
        attrs = [{"timestamp": "2026-01-01"}, {"timestamp": "2026-02-01"}]
        result = trim_evidence(["old", "new"], [(0, 1)], strategy="keep_recent", items_attrs=attrs)
        self.assertEqual(result.kept, (1,))
        self.assertEqual(result.dropped[0].index, 0)
        self.assertEqual(result.apply(["old", "new"]), ["new"])

    def test_keep_first_and_last(self) -> None:
        self.assertEqual(trim_evidence(["a", "b"], [(0, 1)], strategy="keep_first").kept, (0,))
        self.assertEqual(trim_evidence(["a", "b"], [(0, 1)], strategy="keep_last").kept, (1,))

    def test_chain_conflict_keeps_stable_survivor(self) -> None:
        # A↔B and B↔C: B loses twice, A and C survive.
        result = trim_evidence(["A", "B", "C"], [(0, 1), (1, 2)], strategy="keep_first")
        self.assertEqual(result.kept, (0, 2))

    def test_priority_strategy(self) -> None:
        attrs = [{"priority": 1}, {"priority": 9}]
        result = trim_evidence(["low", "high"], [(0, 1)], strategy="keep_priority", items_attrs=attrs)
        self.assertEqual(result.kept, (1,))

    def test_out_of_range_conflict_rejected(self) -> None:
        with self.assertRaises(IndexError):
            trim_evidence(["only"], [(0, 5)])

    def test_unknown_strategy_rejected(self) -> None:
        with self.assertRaises(ValueError):
            trim_evidence(["a"], [], strategy="magic")


class FakeServices:
    def embed(self, texts):
        return [(1.0, 0.0) if "light" in t else (0.0, 1.0) for t in texts]

    def nli_label(self, a, b):
        return "neutral"

    def nli_score(self, a, b, label):
        return 0.5


GOOD_PROPOSAL = json.dumps(
    [
        {
            "id": "proposed_action_conflict",
            "if": [
                {"pred": "attr_equal", "attr": "object"},
                {"pred": "attr_conflict", "attr": "action"},
            ],
            "then": {"label": "conflict"},
        },
        {
            "id": "proposed_nonsense",
            "if": [{"pred": "does_not_exist"}],
            "then": {"label": "conflict"},
        },
    ]
)


class DiscoveryTests(unittest.TestCase):
    def _labeled(self):
        positives = [
            (Item("Turn on the light", {"object": "light", "action": "on"}),
             Item("Turn off the light", {"object": "light", "action": "off"})),
            (Item("Open the door", {"object": "door", "action": "open"}),
             Item("Close the door", {"object": "door", "action": "close"})),
        ]
        negatives = [
            (Item("Turn on the light", {"object": "light", "action": "on"}),
             Item("Water the plants", {"object": "plants", "action": "water"})),
            (Item("Feed the cat", {"object": "cat", "action": "feed"}),
             Item("Feed the cat", {"object": "cat", "action": "feed"})),
        ]
        return positives, negatives

    def test_discover_accepts_valid_and_rejects_bogus(self) -> None:
        positives, negatives = self._labeled()
        report = discover_rules(
            positives,
            negatives,
            engine=RuleEngine(services=FakeServices()),
            chat_fn=lambda prompt: GOOD_PROPOSAL,
            min_coverage=0.2,
            min_accuracy=0.8,
        )
        self.assertEqual(report.candidates_total, 2)
        self.assertEqual(report.candidates_accepted, 1)
        rule = report.ruleset.rules[0]
        self.assertEqual(rule.source, "discovered:llm")
        self.assertEqual(rule.then.label, "conflict")
        coverage, accuracy = report.per_rule[0][1], report.per_rule[0][2]
        self.assertEqual((coverage, accuracy), (1.0, 1.0))

    def test_discover_filters_low_accuracy_candidates(self) -> None:
        # A rule that fires on everything has full coverage but 50% accuracy.
        proposal = json.dumps(
            [{"id": "fires_on_all", "if": [{"pred": "attr_present", "attr": "object"}],
              "then": {"label": "conflict"}}]
        )
        positives, negatives = self._labeled()
        report = discover_rules(
            positives,
            negatives,
            engine=RuleEngine(services=FakeServices()),
            chat_fn=lambda prompt: proposal,
            min_coverage=0.2,
            min_accuracy=0.8,
        )
        self.assertEqual(report.candidates_accepted, 0)

    def test_discover_tolerates_prose_around_json(self) -> None:
        positives, negatives = self._labeled()
        report = discover_rules(
            positives,
            negatives,
            engine=RuleEngine(services=FakeServices()),
            chat_fn=lambda prompt: "Here you go:\n" + GOOD_PROPOSAL + "\nHope this helps!",
        )
        self.assertEqual(report.candidates_total, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
