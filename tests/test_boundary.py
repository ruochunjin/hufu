"""Boundary-behavior tests pinned from the H-list review (offline, L0).

- ``outlier_scores`` on all-constant features must yield 0.0 scores, never
  NaN/inf (MAD=0 falls back to std, std=0 then skips the feature).
- ``RuleSet.load`` / ``Rule.from_dict`` tolerate missing optional fields
  (version/metadata/source/description/confidence get defaults) and raise
  ``KeyError`` for structurally missing required fields (``if``/``then``/
  ``id``/``pred``), which callers should treat as a malformed file.
"""

from __future__ import annotations

import json
import math
import unittest

from hufu.rules.model import Rule, RuleSet
from hufu.scenarios.outliers import outlier_scores


class OutlierConstantFeatureTests(unittest.TestCase):
    def test_all_constant_features_score_zero(self) -> None:
        records = [{"a": 5, "b": 1.5} for _ in range(10)]
        scores = outlier_scores(records)
        self.assertEqual(scores, [0.0] * 10)
        for score in scores:
            self.assertTrue(math.isfinite(score))

    def test_constant_column_plus_outlier_column(self) -> None:
        records = [{"const": 7, "x": float(i)} for i in range(20)]
        records.append({"const": 7, "x": 1000.0})
        scores = outlier_scores(records)
        self.assertTrue(all(math.isfinite(s) for s in scores))
        self.assertEqual(scores[-1], max(scores))
        self.assertGreater(scores[-1], 3.0)

    def test_binary_column_never_infinite(self) -> None:
        records = [{"flag": 0} for _ in range(19)] + [{"flag": 1}]
        scores = outlier_scores(records)
        self.assertTrue(all(math.isfinite(s) for s in scores))


class RuleSetLoadToleranceTests(unittest.TestCase):
    def _base_rule(self) -> dict:
        return {"id": "r1", "if": [{"pred": "always"}], "then": {"label": "pass"}}

    def test_optional_fields_get_defaults(self) -> None:
        rule = Rule.from_dict(self._base_rule())
        self.assertEqual(rule.source, "manual")
        self.assertEqual(rule.description, "")
        self.assertEqual(rule.then.confidence, 1.0)

    def test_ruleset_defaults_version_and_metadata(self) -> None:
        rule_set = RuleSet.from_dict({"name": "s", "rules": [self._base_rule()]})
        self.assertEqual(rule_set.version, "1")
        self.assertEqual(rule_set.metadata, ())

    def test_missing_if_raises_keyerror(self) -> None:
        broken = self._base_rule()
        del broken["if"]
        with self.assertRaises(KeyError):
            Rule.from_dict(broken)

    def test_missing_then_raises_keyerror(self) -> None:
        broken = self._base_rule()
        del broken["then"]
        with self.assertRaises(KeyError):
            Rule.from_dict(broken)

    def test_missing_id_raises_keyerror(self) -> None:
        broken = self._base_rule()
        del broken["id"]
        with self.assertRaises(KeyError):
            Rule.from_dict(broken)

    def test_condition_missing_pred_raises_keyerror(self) -> None:
        broken = self._base_rule()
        broken["if"] = [{"threshold": 0.5}]
        with self.assertRaises(KeyError):
            Rule.from_dict(broken)

    def test_load_roundtrip(self) -> None:
        import tempfile
        from pathlib import Path

        rule_set = RuleSet.from_dict(
            {"name": "s", "version": "2", "rules": [self._base_rule()]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            rule_set.dump(path)
            loaded = RuleSet.load(path)
        self.assertEqual(loaded.name, "s")
        self.assertEqual(loaded.version, "2")
        self.assertEqual(len(loaded.rules), 1)

    def test_load_malformed_json_raises(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                RuleSet.load(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
