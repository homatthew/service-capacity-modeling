"""Typed regret summaries and topology aggregation for capacity plans.

**Experimental** — this API may change.

Replaces the legacy ``(plan, desires, total_regret)`` tuple shape with
typed records (``SampledPlan``, ``RegretCandidate``, ``RegretPlanSummary``)
so downstream consumers can interrogate regret without parsing tuples.

Public API:
- ``regret_detailed`` — N² regret math, returns ``RegretCandidate``.
- ``summarize_regret_candidates`` / ``summaries_for_least_regret`` /
  ``considered_alternative_summaries`` — assemble topology summaries.
- ``topology_signature`` — stable topology hash that ignores cost noise.
- ``merge_plan_components`` / ``merge_regret_candidates_positional`` —
  composed-model helpers.
"""

from __future__ import annotations

import functools
import json
from hashlib import blake2b
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set

import numpy as np
from pydantic import ConfigDict

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import SampleRef
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import merge_plan


# Keys stripped from CapacityPlan dumps before hashing for
# topology_signature; cost values are volatile across pricing refreshes
# so the topology hash deliberately ignores them.
_TOPOLOGY_VOLATILE_KEYS = frozenset(
    {
        "rank",
        "annual_cost",
        "annual_costs",
        "annual_cost_per_gib",
        "annual_cost_per_read_io",
        "annual_cost_per_write_io",
        "annual_cost_per_core",
        "annual_cost_override",
        "total_annual_cost",
        "cost_per_vcpu_annual",
    }
)


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v)
            for k, v in obj.items()
            if k not in _TOPOLOGY_VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def topology_signature(plan: CapacityPlan) -> str:
    """Stable 8-byte blake2b hash of a plan's topology, cost-noise free."""
    raw = plan.model_dump(mode="json")
    stripped = _strip_volatile(raw)
    payload = json.dumps(stripped, sort_keys=True, default=str).encode()
    return blake2b(payload, digest_size=8).hexdigest()


class SampledPlan(ExcludeUnsetModel):
    """One simulation outcome: (sample, desires, chosen plan)."""

    sample: SampleRef
    desires: CapacityDesires
    plan: CapacityPlan


class RegretCandidate(ExcludeUnsetModel):
    """One proposed plan with total regret and provenance.

    ``components_by_model`` is keyed by sub-model name; empty when
    ``regret_detailed`` is called without ``model_name`` (legacy shim).
    """

    plan: CapacityPlan
    desires: CapacityDesires
    total_regret: float
    components_by_model: Dict[str, Dict[str, float]] = {}
    sample: SampleRef

    model_config = ConfigDict(protected_namespaces=())


class RegretPlanSummary(ExcludeUnsetModel):
    """Aggregated regret over all candidates sharing one topology."""

    plan: CapacityPlan
    sample_count: int
    mean_total_regret: float
    mean_regret_components_by_model: Dict[str, Dict[str, float]] = {}
    example_samples: List[SampleRef] = []

    model_config = ConfigDict(protected_namespaces=())


def regret_detailed(
    capacity_plans: Sequence[SampledPlan],
    regret_params: CapacityRegretParameters,
    model: CapacityModel,
    model_name: Optional[str] = None,
) -> Sequence[RegretCandidate]:
    """Compute regret-ranked candidates from sampled plans (N\u00b2).

    When ``model_name`` is provided, each candidate's
    ``components_by_model[model_name]`` holds the per-component regret
    sum. Sorted ascending by ``total_regret``.
    """
    if not capacity_plans:
        return []

    n = len(capacity_plans)
    regret = np.zeros(n, dtype=np.float64)
    candidates: List[RegretCandidate] = []
    for i, proposed in enumerate(capacity_plans):
        components_sum: Dict[str, float] = {}
        for j, optimal in enumerate(capacity_plans):
            # Preserve legacy ``_regret`` arithmetic verbatim: the
            # diagonal call still happens so byte-stable baseline hashes
            # don't drift when model.regret() has a non-zero same-plan
            # result.
            if j == i:
                regret[j] = 0
            components = model.regret(
                regret_params=regret_params,
                optimal_plan=optimal.plan,
                proposed_plan=proposed.plan,
            )
            regret[j] = sum(components.values())
            for k, v in components.items():
                components_sum[k] = components_sum.get(k, 0.0) + v
        total = float(np.einsum("i->", regret))
        component_kwargs: Dict[str, Dict[str, float]] = (
            {model_name: components_sum} if model_name else {}
        )
        candidates.append(
            RegretCandidate(
                plan=proposed.plan,
                desires=proposed.desires,
                total_regret=total,
                components_by_model=component_kwargs,
                sample=proposed.sample,
            )
        )

    candidates.sort(key=lambda c: c.total_regret)
    return candidates


def _accumulate_components(
    target: Dict[str, Dict[str, float]],
    source: Dict[str, Dict[str, float]],
    weight: float,
) -> None:
    for mname, comps in source.items():
        bucket = target.setdefault(mname, {})
        for k, v in comps.items():
            bucket[k] = bucket.get(k, 0.0) + v * weight


def summarize_regret_candidates(
    candidates: Sequence[RegretCandidate],
    examples_cap: int = 3,
) -> List[RegretPlanSummary]:
    """Aggregate candidates by topology; ascending by ``mean_total_regret``."""
    by_topo: Dict[str, List[RegretCandidate]] = {}
    first_plan: Dict[str, CapacityPlan] = {}
    for c in candidates:
        sig = topology_signature(c.plan)
        if sig not in by_topo:
            by_topo[sig] = []
            first_plan[sig] = c.plan
        by_topo[sig].append(c)

    summaries: List[RegretPlanSummary] = []
    for sig, group in by_topo.items():
        n = len(group)
        weight = 1.0 / n
        mean_components: Dict[str, Dict[str, float]] = {}
        seen_ids: Set[str] = set()
        examples: List[SampleRef] = []
        for c in group:
            _accumulate_components(mean_components, c.components_by_model, weight)
            if c.sample.sample_id in seen_ids:
                continue
            seen_ids.add(c.sample.sample_id)
            if len(examples) < examples_cap:
                examples.append(c.sample)
        summaries.append(
            RegretPlanSummary(
                plan=first_plan[sig],
                sample_count=n,
                mean_total_regret=sum(c.total_regret for c in group) * weight,
                mean_regret_components_by_model=mean_components,
                example_samples=examples,
            )
        )

    summaries.sort(key=lambda s: s.mean_total_regret)
    return summaries


def summaries_for_least_regret(
    least_regret: Sequence[CapacityPlan],
    candidates: Sequence[RegretCandidate],
) -> List[RegretPlanSummary]:
    """Map selected plans to summaries; raise ``KeyError`` on missing."""
    summary_by_topo: Dict[str, RegretPlanSummary] = {
        topology_signature(s.plan): s for s in summarize_regret_candidates(candidates)
    }
    result: List[RegretPlanSummary] = []
    for plan in least_regret:
        sig = topology_signature(plan)
        if sig not in summary_by_topo:
            raise KeyError(f"No regret summary for topology {sig!r}")
        result.append(summary_by_topo[sig])
    return result


def considered_alternative_summaries(
    candidates: Sequence[RegretCandidate],
    selected_topologies: Set[str],
    cap: int = 10,
) -> List[RegretPlanSummary]:
    """Return non-selected summaries, regret-sorted and capped at ``cap``."""
    alternatives = [
        s
        for s in summarize_regret_candidates(candidates)
        if topology_signature(s.plan) not in selected_topologies
    ]
    return alternatives[:cap]


def merge_plan_components(composed: Sequence[CapacityPlan]) -> CapacityPlan:
    """Reduce sub-model plans via ``merge_plan``; raises on empty/None."""
    if not composed:
        raise ValueError("merge_plan_components requires at least one plan")
    reducer = cast(
        Callable[[CapacityPlan, CapacityPlan], CapacityPlan],
        merge_plan,
    )
    result = functools.reduce(reducer, composed)
    if result is None:
        raise ValueError("merge_plan_components reduced to None")
    return result


def merge_regret_candidates_positional(
    per_model: Sequence[Sequence[RegretCandidate]],
) -> Sequence[RegretCandidate]:
    """Zip-merge per-sub-model candidate lists; raises on count mismatch.

    Sample/desires are taken from the first sub-model at each index.
    """
    if not per_model:
        return []
    counts = {len(c) for c in per_model}
    if len(counts) > 1:
        raise ValueError(
            f"merge_regret_candidates_positional: sub-model candidate "
            f"counts differ: {sorted(counts)}"
        )

    merged: List[RegretCandidate] = []
    for ith in zip(*per_model):
        plan = merge_plan_components([c.plan for c in ith])
        components: Dict[str, Dict[str, float]] = {}
        for c in ith:
            for mname, comps in c.components_by_model.items():
                components[mname] = dict(comps)
        merged.append(
            RegretCandidate(
                plan=plan,
                desires=ith[0].desires,
                total_regret=sum(c.total_regret for c in ith),
                components_by_model=components,
                sample=ith[0].sample,
            )
        )
    return merged
