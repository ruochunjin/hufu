"""Immutable configuration contracts for HyperMatch algorithms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StandardHyperMatchConfig:
    """Frozen controls for bounded seed expansion and model thresholds.

    Bounded seed expansion has no structural-beam parameter, so this contract
    exposes only the controls consumed by that candidate generator.
    """

    method_id: str = "hypermatch_standard"
    mode: str = "standard"
    fancy: bool = False
    runtime_backend: str = "component_standard"
    hccalc_algorithm: str = "structural_seed_expansion"
    seed_edges_per_query_context: int = 8
    max_seed_expansions: int = 128
    max_cluster_size: int = 5
    max_candidates_per_query_context: int = 128
    hc_backend: str = "qwen3-embedding-0.6b-three-view-current"
    hc_threshold: float = 0.45
    hc_destructive_threshold: float = 0.80
    dmatch_backend: str = "prompted_hybrid"
    dmatch_gnn_weight: float = 0.20
    dmatch_threshold: float = 0.6348974168300628

    @property
    def fingerprint(self) -> str:
        """Return the stable fingerprint of this configuration."""

        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class HCCalcConfig:
    """Bound the four deterministic structural proposal routes.

    ``seed_edges`` limits the strongest singleton seeds per query edge, while
    ``structural_beam`` limits multi-edge expansions.  These are implementation
    budgets, not a relabelling of the top-B endpoint-match bound;
    ``endpoint_match_limit`` controls that independent route.
    """

    proposal_routes: tuple[str, ...] = (
        "relation_seeded",
        "assignment_cover",
        "h_v_endpoint_shortest_path",
        "father_atom_incidence_closure",
    )
    seed_edges: int = 3
    structural_beam: int = 7
    max_cluster_size: int = 5
    closure_max_cluster_size: int = 12
    singleton_shortlist: int = 6
    multi_edge_shortlist: int = 8
    closure_shortlist: int = 24
    endpoint_match_limit: int = 3
    choices_per_query_edge: int = 2
    max_assignment_combinations: int = 64
    query_coverage_weight: float = 0.75
    shortlist_strategy: str = "score_head_with_route_quotas"
    multi_edge_size_quotas: tuple[tuple[int, int], ...] = (
        (2, 3),
        (3, 2),
        (4, 1),
        (5, 1),
    )

    def __post_init__(self) -> None:
        positive = (
            self.seed_edges,
            self.structural_beam,
            self.max_cluster_size,
            self.closure_max_cluster_size,
            self.singleton_shortlist,
            self.multi_edge_shortlist,
            self.closure_shortlist,
            self.endpoint_match_limit,
            self.choices_per_query_edge,
            self.max_assignment_combinations,
        )
        if not all(value > 0 for value in positive):
            raise ValueError("all HCCalc budgets must be positive")
        if self.closure_max_cluster_size < self.max_cluster_size:
            raise ValueError("father-atom closure cap cannot be smaller than m")
        if not 0.0 <= self.query_coverage_weight <= 1.0:
            raise ValueError("query coverage weight must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class HCDecisionConfig:
    """Thresholds that separate rescue, support, and destructive HC use."""

    support_threshold: float = 0.65
    rescue_threshold: float = 0.55
    single_answer_rescue_threshold: float = 0.60
    destructive_threshold: float = 0.85

    def __post_init__(self) -> None:
        values = (
            self.rescue_threshold,
            self.single_answer_rescue_threshold,
            self.support_threshold,
            self.destructive_threshold,
        )
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("HC thresholds must be probabilities")
        if tuple(sorted(values)) != values:
            raise ValueError("HC thresholds must be monotonically increasing")


@dataclass(frozen=True, slots=True)
class DMatchConfig:
    """Role-decoding thresholds and the weights of its two rendered views."""

    ordinary_role_threshold: float = 0.45989492535591125
    destructive_role_threshold: float = 0.6348974168300628
    masked_view_weight: float = 0.5
    target_surface_view_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.ordinary_role_threshold <= 1.0:
            raise ValueError("D-match threshold must be a probability")
        if not self.ordinary_role_threshold <= self.destructive_role_threshold <= 1.0:
            raise ValueError("destructive D-match threshold is invalid")
        if abs(self.masked_view_weight + self.target_surface_view_weight - 1.0) > 1e-9:
            raise ValueError("D-match view weights must sum to one")


@dataclass(frozen=True, slots=True)
class FancyHyperMatchConfig:
    """Immutable controls for multi-route HCCalc and constrained scoring."""

    method_id: str = "hypermatch_fancy"
    hccalc: HCCalcConfig = HCCalcConfig()
    hc: HCDecisionConfig = HCDecisionConfig()
    dmatch: DMatchConfig = DMatchConfig()
    three_view_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    render_schema: str = "masked-nary-role-structure"
    hc_prompt_id: str = "masked-comparable-fact"
    dmatch_prompt_id: str = "canonical-semantic-role"
    fixed_point_mode: str = "all_hc"

    @property
    def mode(self) -> str:
        """Return the externally serialized mode label."""

        return "fancy"

    @property
    def fancy(self) -> bool:
        """Return whether multi-route structural proposals are enabled."""

        return True

    def __post_init__(self) -> None:
        if abs(sum(self.three_view_weights) - 1.0) > 1e-9:
            raise ValueError("HC view weights must sum to one")
        if self.fixed_point_mode != "all_hc":
            raise ValueError("the fancy HyperMatch profile uses all-HC quantification")

    @property
    def fingerprint(self) -> str:
        """Return the stable fingerprint of this configuration."""

        return _fingerprint(asdict(self))


FANCY_CONFIG = FancyHyperMatchConfig()
STANDARD_CONFIG = StandardHyperMatchConfig()


def config_for_mode(
    *, fancy: bool
) -> FancyHyperMatchConfig | StandardHyperMatchConfig:
    """Return the immutable configuration selected by the public option."""

    return FANCY_CONFIG if fancy else STANDARD_CONFIG


def config_summary(
    config: FancyHyperMatchConfig | StandardHyperMatchConfig,
) -> dict[str, Any]:
    """Return a JSON-safe description without importing either model stack."""

    payload = asdict(config)
    payload.update(
        {
            "mode": config.mode,
            "fancy": config.fancy,
            "fingerprint": config.fingerprint,
        }
    )
    if isinstance(config, FancyHyperMatchConfig):
        payload["runtime_backend"] = "model_injected_component"
        payload["entrypoint"] = (
            "hufu.engine.hyper_simulation.run_hypermatch"
        )
    else:
        payload["entrypoint"] = (
            "hufu.engine.hyper_simulation.compute_hyper_simulation"
        )
        # Bounded seed expansion has no structural-beam parameter.
        payload["structural_beam_B"] = None
    return payload


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DMatchConfig",
    "FANCY_CONFIG",
    "HCDecisionConfig",
    "HCCalcConfig",
    "FancyHyperMatchConfig",
    "STANDARD_CONFIG",
    "StandardHyperMatchConfig",
    "config_for_mode",
    "config_summary",
]
