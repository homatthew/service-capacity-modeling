"""Tests for the regret_explainability module.

Covers the typed regret-candidate API extracted from
``capacity_planner._regret`` and topology-aware summary helpers.
"""

from decimal import Decimal
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pytest

from service_capacity_modeling.interface import (
    CapacityDesires,
    CapacityPlan,
    CapacityRegretParameters,
    Clusters,
    DataShape,
    Instance,
    QueryPattern,
    Requirements,
    SampleRef,
    ZoneClusterCapacity,
    certain_int,
)
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.regret_explainability import (
    RegretCandidate,
    SampledPlan,
    considered_alternative_summaries,
    merge_plan_components,
    merge_regret_candidates_positional,
    regret_detailed,
    summaries_for_least_regret,
    summarize_regret_candidates,
    topology_signature,
)


def _ref(label: str = "x") -> SampleRef:
    return SampleRef(sample_id=f"s-0000-{label}", sample_label=label)


def _desires(state_gib: int = 0) -> CapacityDesires:
    return CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(0),
            estimated_write_per_second=certain_int(0),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(state_gib)),
    )


def _plan(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    *,
    instance_name: str = "r6a.xlarge",
    count: int = 3,
    annual_cost: float = 1000.0,
    cost: float = 3000.0,
    rank: float = 0.0,
) -> CapacityPlan:
    instance = Instance(
        name=instance_name,
        cpu=4,
        cpu_ghz=2.5,
        ram_gib=32.0,
        net_mbps=2000,
        drive=None,
        annual_cost=annual_cost,
    )
    cluster = ZoneClusterCapacity(
        cluster_type="test",
        count=count,
        instance=instance,
        attached_drives=[],
    )
    return CapacityPlan(
        requirements=Requirements(),
        candidate_clusters=Clusters(
            annual_costs={"test": Decimal(str(cost))},
            zonal=[cluster],
        ),
        rank=rank,
    )


def _candidate(
    *,
    instance_name: str,
    total_regret: float,
    sample_id: str = "s-0001-a",
) -> RegretCandidate:
    return RegretCandidate(
        plan=_plan(instance_name=instance_name),
        desires=_desires(),
        total_regret=total_regret,
        sample=SampleRef(sample_id=sample_id, sample_label="x"),
    )


class _ConstantRegretModel(CapacityModel):
    """Pins ``model.regret`` to a lookup table so tests can fix the
    ascending-regret ordering without depending on Cassandra math."""

    def __init__(self, table: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        super().__init__()
        self.table = table or {}

    def regret(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        regret_params: CapacityRegretParameters,
        optimal_plan: CapacityPlan,
        proposed_plan: CapacityPlan,
    ) -> Dict[str, float]:
        _ = regret_params
        key = (
            proposed_plan.candidate_clusters.zonal[0].instance.name,
            optimal_plan.candidate_clusters.zonal[0].instance.name,
        )
        return self.table.get(":".join(key), {"spend": 0.0})


def test_regret_detailed_returns_typed_candidates():
    sampled = [
        SampledPlan(
            sample=_ref(f"r{i}"),
            desires=_desires(),
            plan=_plan(instance_name=f"r6a.{i}xlarge"),
        )
        for i in range(3)
    ]
    candidates = regret_detailed(
        sampled, CapacityRegretParameters(), _ConstantRegretModel()
    )
    assert len(candidates) == 3
    assert all(isinstance(c, RegretCandidate) for c in candidates)


def test_regret_detailed_preserves_legacy_order():
    # 'a' totals ≤ 'c' totals ≤ 'b' totals when summed across optimals.
    table = {
        "a:b": {"spend": 1.0},
        "a:c": {"spend": 1.0},
        "b:a": {"spend": 10.0},
        "b:c": {"spend": 10.0},
        "c:a": {"spend": 4.0},
        "c:b": {"spend": 4.0},
    }
    sampled = [
        SampledPlan(sample=_ref(n), desires=_desires(), plan=_plan(instance_name=n))
        for n in ("a", "b", "c")
    ]
    candidates = regret_detailed(
        sampled, CapacityRegretParameters(), _ConstantRegretModel(table)
    )
    instances = [c.plan.candidate_clusters.zonal[0].instance.name for c in candidates]
    assert instances == ["a", "c", "b"]
    assert [c.total_regret for c in candidates] == sorted(
        c.total_regret for c in candidates
    )


def test_topology_signature_stable_across_cost_noise():
    plan_a = _plan(annual_cost=1000.0, cost=3000.0, rank=0.0)
    plan_b = _plan(annual_cost=9999.99, cost=42_424.0, rank=99.5)
    assert topology_signature(plan_a) == topology_signature(plan_b)
    # And different topologies hash differently.
    assert topology_signature(_plan(instance_name="r6a.4xlarge")) != (
        topology_signature(_plan(instance_name="r6a.xlarge"))
    )


def test_summarize_regret_candidates_groups_by_topology():
    candidates = [
        _candidate(instance_name="r6a.xlarge", total_regret=10.0, sample_id="s-1"),
        _candidate(instance_name="r6a.xlarge", total_regret=20.0, sample_id="s-2"),
        _candidate(instance_name="r6a.4xlarge", total_regret=5.0, sample_id="s-3"),
    ]
    summaries = summarize_regret_candidates(candidates)
    by_inst = {s.plan.candidate_clusters.zonal[0].instance.name: s for s in summaries}
    assert by_inst["r6a.xlarge"].sample_count == 2
    assert by_inst["r6a.xlarge"].mean_total_regret == pytest.approx(15.0)
    assert by_inst["r6a.4xlarge"].sample_count == 1
    regrets = [s.mean_total_regret for s in summaries]
    assert regrets == sorted(regrets)


def test_summaries_for_least_regret_raises_on_missing():
    candidates = [_candidate(instance_name="r6a.xlarge", total_regret=10.0)]
    with pytest.raises(KeyError):
        summaries_for_least_regret([_plan(instance_name="m6a.2xlarge")], candidates)


def test_summaries_for_least_regret_maps_selected_plans():
    candidates = [
        _candidate(instance_name="r6a.xlarge", total_regret=10.0),
        _candidate(instance_name="r6a.4xlarge", total_regret=5.0),
    ]
    summaries = summaries_for_least_regret(
        [_plan(instance_name="r6a.4xlarge")], candidates
    )
    assert summaries[0].plan.candidate_clusters.zonal[0].instance.name == "r6a.4xlarge"


def test_considered_alternatives_capped_sorted_deduped():
    candidates = [
        _candidate(
            instance_name=f"r6a.{i}xlarge",
            total_regret=float(i),
            sample_id=f"s-{i}",
        )
        for i in range(1, 13)
    ] + [_candidate(instance_name="r6a.2xlarge", total_regret=2.0, sample_id="s-dup")]
    selected = topology_signature(_plan(instance_name="r6a.1xlarge"))
    alternatives = considered_alternative_summaries(candidates, {selected}, cap=5)
    assert len(alternatives) == 5
    regrets = [a.mean_total_regret for a in alternatives]
    assert regrets == sorted(regrets)
    names = [a.plan.candidate_clusters.zonal[0].instance.name for a in alternatives]
    assert "r6a.1xlarge" not in names
    # The duplicate r6a.2xlarge collapsed into one summary.
    assert names.count("r6a.2xlarge") == 1


def test_merge_regret_candidates_positional_raises_on_count_mismatch():
    a = [_candidate(instance_name="r6a.xlarge", total_regret=1.0)]
    b: List[RegretCandidate] = []
    with pytest.raises(ValueError):
        merge_regret_candidates_positional([a, b])


def test_merge_regret_candidates_positional_sums_total_regret():
    sub1 = [
        _candidate(instance_name="r6a.xlarge", total_regret=10.0),
        _candidate(instance_name="r6a.4xlarge", total_regret=2.0),
    ]
    sub2 = [
        _candidate(instance_name="r6a.xlarge", total_regret=1.0),
        _candidate(instance_name="r6a.4xlarge", total_regret=3.0),
    ]
    merged = merge_regret_candidates_positional([sub1, sub2])
    assert [c.total_regret for c in merged] == [11.0, 5.0]


def test_merge_plan_components_raises_on_empty():
    with pytest.raises(ValueError):
        merge_plan_components([])


def test_legacy_regret_shim_returns_3tuples():
    """The legacy ``_regret`` shim continues to return ``(plan, desires,
    total)`` tuples after the refactor — required by
    ``ComposedExplanation.regret_clusters`` and consumers like
    ``test_compositional``."""
    # pylint: disable=import-outside-toplevel
    from service_capacity_modeling.capacity_planner import _regret

    capacity_plans: List[Any] = [
        (_desires(state_gib=i), _plan(instance_name=f"r6a.{i}xlarge")) for i in range(3)
    ]
    result = _regret(capacity_plans, CapacityRegretParameters(), _ConstantRegretModel())
    assert all(isinstance(t, tuple) and len(t) == 3 for t in result)
    assert all(isinstance(t[0], CapacityPlan) for t in result)
