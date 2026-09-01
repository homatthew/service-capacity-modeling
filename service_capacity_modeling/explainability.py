"""Explainability types for the capacity planner.

**Experimental** — this API may change.

This module contains the family graph (FamilyTrait, FamilyEdge, FamilyGraph)
and ExplainedPlans — types used to explain *why* the planner rejected
certain instance/drive combinations and what alternatives exist.

Core contract types (Bottleneck, Excuse) live in interface.py because they
are part of the CapacityModel.capacity_plan() return type.

Consumer usage::

    from service_capacity_modeling.capacity_planner import planner
    from service_capacity_modeling.models.plan_comparison import compare_plans

    # Rejection explanations + family graph
    explained = planner.plan_certain_explained(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra,
    )

    # Current-vs-recommended comparison (separate concern)
    baseline = planner.extract_baseline_plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra,
    )
    comparison = compare_plans(baseline, explained.plans[0])

    # Serialize both for downstream consumers
    explained.model_dump_json()
    comparison.model_dump_json()
"""

from __future__ import annotations

import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING

from pydantic import ConfigDict
from pydantic import SerializeAsAny

from service_capacity_modeling.enum_utils import enum_docstrings
from service_capacity_modeling.enum_utils import StrEnum
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import ComposedExplanation
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import ExcuseTag
from service_capacity_modeling.interface import Hardware
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import UncertainCapacityPlan

# regret_explainability imports from models, which imports from
# explainability — we'd hit a circular import if we imported
# RegretPlanSummary at module load. Defer to TYPE_CHECKING and let
# pydantic resolve the forward reference when ``model_rebuild`` is
# called at the bottom of this module.
if TYPE_CHECKING:
    from service_capacity_modeling.regret_explainability import (
        RegretPlanSummary,
    )


class FamilyTrait(ExcludeUnsetModel):
    """Intrinsic hardware properties of an instance family.

    All numeric values are derived from hardware shapes data via
    from_instance(). Within an AWS family, ratios (ram/vcpu, disk/vcpu,
    cost/vcpu) are constant across instance sizes due to linear pricing.
    """

    family: str
    memory_gib_per_vcpu: float
    has_local_disk: bool
    local_disk_gib_per_vcpu: Optional[float] = None
    drive_type: Optional[DriveType] = None
    cost_per_vcpu_annual: Optional[float] = None
    """Annual cost per vCPU derived from loaded pricing (may be internal)."""
    context: Dict[str, Any] = {}
    """Optional model-specific trait extras (populated via the
    `derive_traits` hook passed into FamilyGraph.build)."""

    @classmethod
    def from_instance(cls, instance: Instance) -> FamilyTrait:
        """Derive family traits from any instance in the family.

        Since ratios are constant within a family, any size works.
        Cost uses whatever pricing is loaded (public or internal).
        """
        drive = instance.drive
        has_local = drive is not None
        return cls(
            family=instance.family,
            memory_gib_per_vcpu=round(instance.ram_gib / instance.cpu, 2),
            has_local_disk=has_local,
            local_disk_gib_per_vcpu=(
                round(drive.size_gib / instance.cpu, 1)
                if drive is not None and drive.size_gib > 0
                else None
            ),
            drive_type=drive.drive_type if drive is not None else None,
            cost_per_vcpu_annual=(
                round(instance.annual_cost / instance.cpu, 2)
                if instance.annual_cost and instance.cpu
                else None
            ),
        )


class FamilyEdge(ExcludeUnsetModel):
    """A directed trade-off edge between two instance families.

    Edges encode hardware topology: what improves and what degrades
    when switching from one family to another. All attributes are
    derived at runtime from FamilyTrait hardware data by FamilyGraph.build().
    """

    from_family: str
    to_family: str
    trade_off: Optional[str] = None
    improves: List[Bottleneck] = []
    degrades: List[Bottleneck] = []
    context: Dict[str, Any] = {}
    """Optional model-specific edge extras (populated via the
    `derive_edge` hook passed into FamilyGraph.build)."""


def _family_generation(family: str) -> Optional[int]:
    """Extract generation number from AWS family name (e.g. 'r7a' → 7)."""
    m = re.search(r"\d+", family)
    return int(m.group()) if m else None


def derive_edge_attributes(  # noqa: C901  # pylint: disable=too-many-branches
    from_trait: FamilyTrait,
    to_trait: FamilyTrait,
) -> Tuple[List[Bottleneck], List[Bottleneck]]:
    """Derive improves/degrades for a family pair from their hardware traits."""
    improves: List[Bottleneck] = []
    degrades: List[Bottleneck] = []

    # cost — from loaded pricing (may be internal)
    if from_trait.cost_per_vcpu_annual and to_trait.cost_per_vcpu_annual:
        if to_trait.cost_per_vcpu_annual < from_trait.cost_per_vcpu_annual:
            improves.append(Bottleneck.cost)
        elif to_trait.cost_per_vcpu_annual > from_trait.cost_per_vcpu_annual:
            degrades.append(Bottleneck.cost)

    # memory
    if to_trait.memory_gib_per_vcpu > from_trait.memory_gib_per_vcpu:
        improves.append(Bottleneck.memory)
    elif to_trait.memory_gib_per_vcpu < from_trait.memory_gib_per_vcpu:
        degrades.append(Bottleneck.memory)

    # disk_capacity — local vs EBS
    if from_trait.has_local_disk and to_trait.has_local_disk:
        from_disk = from_trait.local_disk_gib_per_vcpu or 0
        to_disk = to_trait.local_disk_gib_per_vcpu or 0
        if to_disk > from_disk:
            improves.append(Bottleneck.disk_capacity)
        elif to_disk < from_disk:
            degrades.append(Bottleneck.disk_capacity)
    elif from_trait.has_local_disk and not to_trait.has_local_disk:
        improves.append(Bottleneck.disk_capacity)  # EBS: flexible sizing
    elif not from_trait.has_local_disk and to_trait.has_local_disk:
        degrades.append(Bottleneck.disk_capacity)  # local: fixed size

    # disk_iops — local NVMe vs EBS is a qualitative change (latency curve,
    # not just peak IOPS). EBS-to-EBS and local-to-local are omitted: max
    # IOPS is rarely the bottleneck and latency curves are modeled identically.
    if from_trait.has_local_disk and not to_trait.has_local_disk:
        degrades.append(Bottleneck.disk_iops)
    elif not from_trait.has_local_disk and to_trait.has_local_disk:
        improves.append(Bottleneck.disk_iops)

    # generation — derived from family name (r7a=7, r6a=6, i4i=4, ...)
    from_gen = _family_generation(from_trait.family)
    to_gen = _family_generation(to_trait.family)
    if from_gen is not None and to_gen is not None:
        if to_gen > from_gen:
            improves.append(Bottleneck.generation)
        elif to_gen < from_gen:
            degrades.append(Bottleneck.generation)

    return improves, degrades


class FamilyGraph(ExcludeUnsetModel):
    """Soft DAG of instance family trade-off relationships.

    Nodes are FamilyTraits (derived from hardware shapes).
    Edges are FamilyEdges (derived from trait comparisons).
    """

    traits: Dict[str, FamilyTrait] = {}
    edges: List[FamilyEdge] = []

    def suggest_alternatives(self, excuse: Excuse) -> List[FamilyEdge]:
        """Return edges that improve at least one of the excuse's bottlenecks.

        If multiple bottlenecks bind, an edge is suggested when it improves
        any of them. Consumers can re-rank by `len(set(excuse.bottlenecks) &
        set(edge.improves))` if they want.
        """
        if not excuse.bottlenecks:
            return []
        excuse_family = excuse.instance.rsplit(".", 1)[0]
        wanted = set(excuse.bottlenecks)
        return [
            e
            for e in self.edges
            if e.from_family == excuse_family and not wanted.isdisjoint(e.improves)
        ]

    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-positional-arguments
    @classmethod
    def build(  # noqa: C901
        cls,
        excuses: Sequence[Excuse],
        hardware: Hardware,
        preferred_families: Optional[FrozenSet[str]],
        derive_traits: Optional[Callable[[str, Instance], Dict[str, Any]]] = None,
        derive_edge: Optional[
            Callable[
                [FamilyTrait, FamilyTrait, List[Bottleneck], List[Bottleneck]],
                Tuple[List[Bottleneck], List[Bottleneck], Dict[str, Any]],
            ]
        ] = None,
    ) -> FamilyGraph:
        """Build an M×N FamilyGraph from derived hardware traits.

        All edge attributes (cost, memory, disk_capacity, disk_iops, generation)
        are derived at runtime from FamilyTrait values — no hardcoded trade-off
        tables. The current cluster's family is always included even if it falls
        outside preferred_families, so consumers always see why their current
        shape was rejected.

        The graph is always populated from preferred_families. If preferred_families
        is None (model has no declared preference) the base is empty, so only the
        current cluster's family appears as a node (if current_clusters is set).
        This prevents stateless models from inheriting storage-optimized families.

        Hooks (both optional):

        - `derive_traits(family, instance) -> Dict[str, Any]` augments each
          FamilyTrait's `context` slot with model-specific metadata. Called
          once per family.

        - `derive_edge(from_trait, to_trait, default_improves, default_degrades)
           -> (improves, degrades, context)` is called once per directed edge.
          The framework's default improves/degrades (from `derive_edge_attributes`)
          are passed in; models may transform them, ignore them, or just attach
          edge `context`. Default (no hook): pass through with empty context.
        """
        base = preferred_families if preferred_families is not None else frozenset()
        current_shape_families: Set[str] = {
            e.instance.rsplit(".", 1)[0]
            for e in excuses
            if ExcuseTag.current_shape in e.tags
        }
        included = base | current_shape_families

        # Index one instance per family — O(M) single pass
        family_first: Dict[str, Any] = {}
        for inst in hardware.instances.values():
            fam = inst.family
            if fam in included and fam not in family_first:
                family_first[fam] = inst

        traits: Dict[str, FamilyTrait] = {}
        for fam in included:
            if fam in family_first:
                inst = family_first[fam]
                trait = FamilyTrait.from_instance(inst)
                if derive_traits is not None:
                    extra = derive_traits(fam, inst)
                    if extra:
                        trait = trait.model_copy(update={"context": extra})
                traits[fam] = trait

        # M×N directed edges — all pairs, attributes fully derived
        edges: List[FamilyEdge] = []
        for from_fam, from_trait in traits.items():
            for to_fam, to_trait in traits.items():
                if from_fam == to_fam:
                    continue
                default_improves, default_degrades = derive_edge_attributes(
                    from_trait, to_trait
                )
                if derive_edge is not None:
                    improves, degrades, context = derive_edge(
                        from_trait, to_trait, default_improves, default_degrades
                    )
                else:
                    improves, degrades, context = (
                        default_improves,
                        default_degrades,
                        {},
                    )
                edge_kwargs: Dict[str, Any] = {
                    "from_family": from_fam,
                    "to_family": to_fam,
                    "improves": improves,
                    "degrades": degrades,
                }
                if context:
                    edge_kwargs["context"] = context
                edges.append(FamilyEdge(**edge_kwargs))

        return cls(traits=traits, edges=edges)


# Default preferred family set for stateful datastores (Cassandra, Kafka, EVCache).
# Covers the full decision space: one representative per
# {memory-class × storage-class × generation-tier}.
# Not appropriate for stateless services (DGW, Java apps) — those models
# should override preferred_families() with compute/general families only.
# "n"-suffix (enhanced-network) families are intentionally excluded.
STATEFUL_DATASTORE_FAMILIES: FrozenSet[str] = frozenset(
    {
        "c6a",
        "c7a",  # compute-optimized EBS (~1.9 GiB/vCPU)
        "m6a",
        "m7a",  # general-purpose EBS (~3.8 GiB/vCPU)
        "m6id",  # general-purpose local NVMe (~3.8 GiB/vCPU)
        "r6a",
        "r7a",  # memory-optimized EBS (~7.6 GiB/vCPU)
        "r5d",
        "r6id",  # memory-optimized local NVMe (~7.6–8.0 GiB/vCPU)
        "i4i",
        "i3en",  # storage-optimized local NVMe
    }
)

# Preferred family set for stateless services (DGW, Java apps, NodeQuark).
# EBS-only: no local NVMe (storage density is irrelevant), no storage-optimized
# families (i4i/i3en). Covers the compute × memory × generation axes.
# Models override preferred_families() to return this set.
STATELESS_SERVICE_FAMILIES: FrozenSet[str] = frozenset(
    {
        "c6a",
        "c7a",  # compute-optimized EBS (~1.9 GiB/vCPU) — CPU-bound services
        "m6a",
        "m7a",  # general-purpose EBS (~3.8 GiB/vCPU) — balanced workloads
        "r6a",
        "r7a",  # memory-optimized EBS (~7.6 GiB/vCPU) — heap-heavy services
    }
)


@enum_docstrings
class FamilyPreset(StrEnum):
    """Named preset for `PreferredFamilies.preset`."""

    stateful_datastore = "stateful_datastore"
    """Equivalent to STATEFUL_DATASTORE_FAMILIES."""

    stateless_service = "stateless_service"
    """Equivalent to STATELESS_SERVICE_FAMILIES."""


_PRESETS: Dict[FamilyPreset, FrozenSet[str]] = {
    FamilyPreset.stateful_datastore: STATEFUL_DATASTORE_FAMILIES,
    FamilyPreset.stateless_service: STATELESS_SERVICE_FAMILIES,
}


class PreferredFamilies(ExcludeUnsetModel):
    """Composable family preference for `CapacityModel.preferred_families()`.

    Resolution: ``(preset ∪ add) − remove``. If all three are unset,
    ``resolve()`` returns ``None`` ("no preference") — same semantics as a
    model returning ``None`` from ``preferred_families()``. Otherwise
    ``resolve()`` returns a concrete ``FrozenSet[str]``, which may be empty
    (e.g. ``remove`` drains the preset). Empty set means "preferred set is
    empty"; ``FamilyGraph`` still includes the current cluster's family.
    """

    preset: Optional[FamilyPreset] = None
    add: FrozenSet[str] = frozenset()
    remove: FrozenSet[str] = frozenset()

    def resolve(self) -> Optional[FrozenSet[str]]:
        if self.preset is None and not self.add and not self.remove:
            return None
        base = _PRESETS.get(self.preset, frozenset()) if self.preset else frozenset()
        return (base | self.add) - self.remove


MAX_EXAMPLE_SAMPLES: int = 3
"""Cap on ``Excuse.example_samples`` length after dedup. Three is enough
to show variability while keeping JSON output small."""


def _excuse_dedup_key(exc: Excuse) -> Tuple[Any, ...]:
    return (
        exc.instance,
        exc.drive,
        exc.reason,
        tuple(exc.bottlenecks),
        tuple(sorted(exc.tags, key=lambda t: t.value)),
    )


def _finalize_aggregated_excuse(
    first: Excuse,
    total: int,
    samples: List[Any],
    distinct: int,
) -> Excuse:
    update: Dict[str, Any] = {}
    if total != 1:
        update["frequency"] = total
    if samples:
        update["example_samples"] = samples
        if distinct > 1:
            update["sample_count"] = distinct
    if not update:
        return first
    return first.model_copy(update=update)


def deduplicate_excuses(excuses: Sequence[Excuse]) -> Sequence[Excuse]:
    """Aggregate excuses across simulations by counting occurrences.

    Identity key is ``(instance, drive, reason, tuple(bottlenecks),
    tuple(sorted(tags by .value)))``. ``bottlenecks`` is an ordered tuple
    because order encodes priority (most-binding first) — ``[cpu, memory]``
    and ``[memory, cpu]`` are distinct rows. Tags are an unordered set
    within the key so ``[A, B]`` and ``[B, A]`` collide. For every collision
    (including any within a single pass) the aggregator sums
    ``exc.frequency`` so that already-aggregated inputs accumulate
    correctly.

    Sample provenance is aggregated alongside frequency: per key the
    aggregator collects the distinct ``SampleRef`` ids from
    ``exc.example_samples`` capped at ``MAX_EXAMPLE_SAMPLES``, and counts
    the total distinct sample ids into ``sample_count``. Excuses that
    carry no example samples (e.g. emitted by ``plan_certain_explained``)
    leave both fields unset so ``ExcludeUnsetModel`` omits them from
    JSON.

    A singleton (aggregated total of 1) is returned as the first-seen
    ``Excuse`` *unchanged* so its ``frequency`` field stays unset and
    ``ExcludeUnsetModel`` can omit it from JSON. When the total exceeds
    1 the first-seen excuse is copied with the aggregated count via
    ``model_copy(update={"frequency": total})`` so the field becomes set.
    """
    order: List[Tuple[Any, ...]] = []
    first_by_key: Dict[Tuple[Any, ...], Excuse] = {}
    totals: Dict[Tuple[Any, ...], int] = {}
    samples_by_key: Dict[Tuple[Any, ...], List[Any]] = {}
    sample_ids_by_key: Dict[Tuple[Any, ...], Set[str]] = {}
    for exc in excuses:
        key = _excuse_dedup_key(exc)
        if key not in first_by_key:
            first_by_key[key] = exc
            totals[key] = exc.frequency
            samples_by_key[key] = []
            sample_ids_by_key[key] = set()
            order.append(key)
        else:
            totals[key] += exc.frequency
        for sample in exc.example_samples:
            if sample.sample_id in sample_ids_by_key[key]:
                continue
            sample_ids_by_key[key].add(sample.sample_id)
            if len(samples_by_key[key]) < MAX_EXAMPLE_SAMPLES:
                samples_by_key[key].append(sample)

    return [
        _finalize_aggregated_excuse(
            first_by_key[k],
            totals[k],
            samples_by_key[k],
            len(sample_ids_by_key[k]),
        )
        for k in order
    ]


class ModelExplanation(ExcludeUnsetModel):
    """Base for model-specific explanation payloads.

    Subclass to attach typed reasoning to a CapacityPlan. The base form
    carries only ``model_name`` and a free-form ``context`` bag.
    Subclasses add typed fields, e.g.::

        class CassandraExplanation(ModelExplanation):
            rf_choice_reason: str
            heap_sizing_reason: str
            page_cache_gib: float

    Stored on ``ExplainedPlans.model_explanations`` keyed by registered
    model name; consumers read by name and downcast to the expected
    subclass shape. Pydantic v2 serializes the concrete subclass's
    fields, so model-specific keys round-trip in JSON.
    """

    model_name: str
    context: Dict[str, Any] = {}

    # Pydantic v2 warns about the field named `model_name` colliding with
    # its `model_*` protected namespace; we explicitly opt out.
    model_config = ConfigDict(protected_namespaces=())


class ExplainedPlans(ExcludeUnsetModel):
    """Plans + excuses + family context.

    Structured data for programmatic consumers. Serialize with
    .model_dump() / .model_dump_json().
    """

    plans: Sequence[CapacityPlan]
    excuses: Sequence[Excuse] = []
    family_graph: FamilyGraph = FamilyGraph()
    model_explanations: Dict[str, SerializeAsAny[ModelExplanation]] = {}
    """Per-sub-model typed explanation payloads, keyed by registered
    model name. Empty when no sub-model returned an explanation.

    ``SerializeAsAny`` forces pydantic to serialize the concrete runtime
    type of each value (a ``ModelExplanation`` subclass) rather than the
    declared base type, so subclass-only fields round-trip in
    ``model_dump`` / ``model_dump_json``."""


def walk_explanations(
    root: Optional[ComposedExplanation],
) -> Iterator[ComposedExplanation]:
    """Pre-order traversal of a ComposedExplanation tree.

    Yields the root first, then each subtree depth-first. Returns
    immediately if root is None. Pre-order matches the iteration order
    of the legacy by-model dicts when there's no compose_with nesting.
    """
    if root is None:
        return
    yield root
    for child in root.children:
        yield from walk_explanations(child)


class ExplainedUncertainPlans(ExcludeUnsetModel):
    """Wrapper around ``UncertainCapacityPlan`` with derived summaries.

    Returned by ``planner.plan_explained(...)``. Mirrors the
    ``ExplainedPlans`` shape but for the uncertain (multi-sim) path:
    surfaces typed regret summaries, considered alternatives, a
    deduplicated excuse summary, the per-region family graph, and any
    per-sub-model ``ModelExplanation`` payloads.
    """

    plan: UncertainCapacityPlan
    least_regret_summaries: List["RegretPlanSummary"] = []
    considered_alternatives: List["RegretPlanSummary"] = []
    excuse_summary: List[Excuse] = []
    family_graph: FamilyGraph = FamilyGraph()
    model_explanations: Dict[str, SerializeAsAny[ModelExplanation]] = {}
    """Per-sub-model typed explanation payloads, keyed by registered
    model name. Empty when no sub-model returned an explanation."""


def _rebuild_explained_uncertain() -> None:
    """Resolve the forward reference to ``RegretPlanSummary`` once the
    ``regret_explainability`` module is importable. Called eagerly so
    consumers get a fully-validated schema; we accept the small
    bootstrap cost on first import."""
    # pylint: disable=import-outside-toplevel
    from service_capacity_modeling.regret_explainability import (
        RegretPlanSummary as _RegretPlanSummary,
    )

    ExplainedUncertainPlans.model_rebuild(
        _types_namespace={"RegretPlanSummary": _RegretPlanSummary}
    )


_rebuild_explained_uncertain()
