"""Unified quality-rule language (rules DSL) and evaluation engine.

Rule sets are JSON data.  Built-in predicates register themselves into
``DEFAULT_REGISTRY`` when this package is imported::

    from hufu.rules import Item, RuleEngine, RuleSet

    rules = RuleSet.load("my_rules.json")
    engine = RuleEngine()
    matches = engine.check_pair(rules, Item("Turn on the light"), Item("Turn off the light"))
"""

from hufu.rules import (
    predicates as _builtin_predicates,  # noqa: F401 - registers built-ins
)
from hufu.rules.calibrate import ThresholdReport, learn_threshold_youden
from hufu.rules.engine import (
    ConditionTrace,
    DefaultServices,
    RuleEngine,
    RuleMatch,
)
from hufu.rules.model import Condition, Rule, RuleSet, Verdict
from hufu.rules.registry import (
    DEFAULT_REGISTRY,
    Item,
    PredicateRegistry,
    Services,
    predicate,
)

__all__ = [
    "Condition",
    "ThresholdReport",
    "ConditionTrace",
    "DEFAULT_REGISTRY",
    "DefaultServices",
    "Item",
    "PredicateRegistry",
    "Rule",
    "RuleEngine",
    "RuleMatch",
    "RuleSet",
    "Services",
    "Verdict",
    "learn_threshold_youden",
    "predicate",
]
