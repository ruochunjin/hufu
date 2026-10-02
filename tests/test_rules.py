"""Dependency-free tests for the unified rule language and memory scenario.

Neural services are faked; no model, network, or Rust extension is needed.
"""

from __future__ import annotations

import unittest

from hufu.rules import (
    Condition,
    Item,
    PredicateRegistry,
    Rule,
    RuleEngine,
    RuleSet,
    Verdict,
    predicate,
)
from hufu.scenarios.memory import default_ruleset, detect_memory_conflicts


class FakeServices:
    """Deterministic stand-ins for the embedding and NLI models."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            # Light-related sentences point one way, everything else another.
            vectors.append((1.0, 0.0) if "light" in text else (0.0, 1.0))
        return vectors

    def nli_label(self, premise, hypothesis):
        return "neutral"

    def nli_score(self, premise, hypothesis, label):
        if label == "contradiction":
            return 0.9 if {premise[-2:], hypothesis[-2:]} == {"on", "ff"} else 0.1
        return 0.5


def make_engine():
    return RuleEngine(services=FakeServices())


class RuleModelTests(unittest.TestCase):
    def test_ruleset_json_roundtrip(self) -> None:
        rule_set = default_ruleset()
        restored = RuleSet.from_dict(rule_set.to_dict())
        self.assertEqual(restored, rule_set)

    def test_unknown_rule_field_becomes_params(self) -> None:
        rule = Rule.from_dict(
            {
                "id": "t",
                "if": [{"pred": "attr_equal", "attr": "object", "negate": True}],
                "then": {"label": "x"},
            }
        )
        condition = rule.if_[0]
        self.assertEqual(condition.predicate, "attr_equal")
        self.assertTrue(condition.negate)
        self.assertEqual(condition.params_dict(), {"attr": "object"})

    def test_builtin_predicates_registered(self) -> None:
        import hufu.rules as rules_pkg

        for name in ("vec_sim", "attr_equal", "attr_conflict", "nli_contradicts"):
            self.assertIn(name, rules_pkg.DEFAULT_REGISTRY)


class RuleEngineTests(unittest.TestCase):
    def test_cdr_action_conflict_fires_with_explanation(self) -> None:
        engine = make_engine()
        rules = default_ruleset()
        a = Item("Turn on the light", attrs={"object": "light", "action": "on"})
        b = Item("Turn off the light", attrs={"object": "light", "action": "off"})
        matches = engine.check_pair(rules, a, b)
        self.assertTrue(matches)
        match = next(m for m in matches if m.rule.id == "cdr.action_conflict")
        self.assertEqual(match.verdict.label, "conflict")
        self.assertIn("attr_equal=1.000", match.explanation)
        self.assertIn("vec_sim=", match.explanation)

    def test_no_conflict_when_objects_differ(self) -> None:
        engine = make_engine()
        a = Item("Turn on the light", attrs={"object": "light", "action": "on"})
        b = Item("Turn off the lamp", attrs={"object": "lamp", "action": "off"})
        matches = engine.check_pair(default_ruleset(), a, b)
        self.assertFalse([m for m in matches if m.rule.id == "cdr.action_conflict"])

    def test_threshold_and_negate(self) -> None:
        engine = make_engine()
        rule = Rule(
            "t",
            (Condition("vec_sim", threshold=0.9, negate=True),),
            Verdict("dissimilar"),
        )
        # Same embedding direction -> sim 1.0 -> negated condition fails.
        a = Item("light a")
        b = Item("light b")
        self.assertIsNone(engine.check_rule(rule, a, b))
        # Opposite directions -> sim -1.0 -> negated condition passes.
        c = Item("nothing here")
        match = engine.check_rule(rule, a, c)
        self.assertIsNotNone(match)
        self.assertEqual(match.verdict.label, "dissimilar")

    def test_numeric_and_temporal_predicates(self) -> None:
        engine = make_engine()
        rules = RuleSet(
            "t",
            (
                Rule(
                    "num",
                    (Condition("numeric_conflict", params=(("attr", "value"), ("rel_tol", 0.05))),),
                    Verdict("conflict"),
                ),
                Rule(
                    "time",
                    (Condition("temporal_order", params=(("attr", "timestamp"),)),),
                    Verdict("ordered"),
                ),
            ),
        )
        a = Item("temp is 20", attrs={"object": "room", "value": 20, "timestamp": "2026-01-01"})
        b = Item("temp is 30", attrs={"object": "room", "value": 30, "timestamp": "2026-02-01"})
        labels = {m.verdict.label for m in engine.check_pair(rules, a, b)}
        self.assertEqual(labels, {"conflict", "ordered"})

    def test_custom_predicate_registration(self) -> None:
        registry = PredicateRegistry()

        @predicate("same_first_letter", registry=registry)
        def same_first_letter(a, b, params, services):
            return 1.0 if a.text[:1] == b.text[:1] else 0.0

        engine = RuleEngine(registry=registry, services=FakeServices())
        rule = Rule("t", (Condition("same_first_letter"),), Verdict("match"))
        self.assertIsNotNone(engine.check_rule(rule, Item("apple"), Item("avocado")))
        self.assertIsNone(engine.check_rule(rule, Item("apple"), Item("banana")))


class MemoryScenarioTests(unittest.TestCase):
    def test_detect_memory_conflicts_end_to_end(self) -> None:
        conflicts = detect_memory_conflicts(
            [
                ("Turn on the light", {"object": "light", "action": "on"}),
                ("Turn off the light", {"object": "light", "action": "off"}),
                ("Water the plants", {"object": "plants", "action": "water"}),
            ],
            engine=make_engine(),
        )
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual((conflict.index_a, conflict.index_b), (0, 1))
        self.assertIn("conflict", conflict.labels)
        self.assertTrue(any("cdr.action_conflict" in e for e in conflict.explanations))

    def test_user_attrs_win_over_llm_enrichment(self) -> None:
        # enrich_with_llm=False by default; attrs pass through untouched.
        conflicts = detect_memory_conflicts(
            [("Door is open", {"object": "door", "state": "open"}),
             ("Door is closed", {"object": "door", "state": "closed"})],
            engine=make_engine(),
        )
        self.assertEqual(len(conflicts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
