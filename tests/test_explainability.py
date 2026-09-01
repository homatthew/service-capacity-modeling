"""Tests for core explainability types: Excuse, FamilyTrait, FamilyEdge, FamilyGraph.

These tests verify library-level contracts with no model-specific logic.
Model integration tests live in tests/netflix/test_<model>_explainability.py.
"""

# pylint: disable=too-many-lines

import pytest

from service_capacity_modeling.capacity_planner import (
    _format_sample_value,
    _sample_ref_for_desires,
    planner,
)
from service_capacity_modeling.explainability import (
    ExplainedPlans,
    FamilyEdge,
    FamilyGraph,
    FamilyPreset,
    FamilyTrait,
    MAX_EXAMPLE_SAMPLES,
    ModelExplanation,
    PreferredFamilies,
    STATEFUL_DATASTORE_FAMILIES,
    STATELESS_SERVICE_FAMILIES,
    deduplicate_excuses,
    derive_edge_attributes,
    walk_explanations,
)
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    Bottleneck,
    CapacityDesires,
    ComposedExplanation,
    DataShape,
    Excuse,
    ExcuseTag,
    QueryPattern,
    SampleRef,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.utils import compute_excuse_tags


class TestExcuseModel:
    """Test the Excuse data model."""

    def test_excuse_basic_fields(self):
        excuse = Excuse(
            instance="r6a.2xlarge",
            drive="gp3",
            reason="Cluster too large: 128 nodes > max 64",
            bottlenecks=[Bottleneck.disk_capacity],
        )
        assert excuse.instance == "r6a.2xlarge"
        assert excuse.drive == "gp3"
        assert excuse.bottlenecks == [Bottleneck.disk_capacity]
        assert not excuse.tags
        assert not excuse.context

    def test_excuse_with_context(self):
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="Requires attached disks but i4i has local drives",
            context={"instance_drive": "local_nvme", "require_attached_disks": True},
            bottlenecks=[Bottleneck.drive_type],
            tags=["different_family"],
        )
        assert excuse.context["require_attached_disks"] is True
        assert "different_family" in excuse.tags

    def test_excuse_serialization_excludes_unset(self):
        excuse = Excuse(
            instance="r6a.xlarge",
            drive="gp3",
            reason="Instance too small",
        )
        data = excuse.model_dump()
        assert "bottlenecks" not in data
        assert data["instance"] == "r6a.xlarge"


class TestFamilyTrait:
    """Test FamilyTrait.from_instance() derivation."""

    def test_from_instance_local_disk(self):
        hardware = shapes.region("us-east-1")
        inst = hardware.instances["i4i.8xlarge"]
        trait = FamilyTrait.from_instance(inst)
        assert trait.family == "i4i"
        assert trait.has_local_disk is True
        assert trait.memory_gib_per_vcpu == pytest.approx(7.63, abs=0.1)
        assert trait.local_disk_gib_per_vcpu is not None
        assert trait.local_disk_gib_per_vcpu > 200

    def test_from_instance_ebs(self):
        hardware = shapes.region("us-east-1")
        r6a_inst = next(
            inst for inst in hardware.instances.values() if inst.family == "r6a"
        )
        trait = FamilyTrait.from_instance(r6a_inst)
        assert trait.family == "r6a"
        assert trait.has_local_disk is False
        assert trait.local_disk_gib_per_vcpu is None
        assert trait.drive_type is None
        assert trait.memory_gib_per_vcpu > 7.0

    def test_ratios_constant_across_sizes(self):
        hardware = shapes.region("us-east-1")
        i4i_traits = [
            FamilyTrait.from_instance(inst)
            for inst in hardware.instances.values()
            if inst.family == "i4i" and inst.cpu >= 4
        ]
        assert len(i4i_traits) >= 2
        ratios = {t.memory_gib_per_vcpu for t in i4i_traits}
        assert len(ratios) == 1


class TestFamilyGraph:
    """Test FamilyGraph.suggest_alternatives."""

    def test_suggest_alternatives_finds_edges(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="i4i",
                    to_family="i3en",
                    trade_off="4x disk/node",
                    improves=[Bottleneck.disk_capacity],
                    degrades=[Bottleneck.disk_iops],
                ),
                FamilyEdge(
                    from_family="i4i",
                    to_family="r7a",
                    trade_off="EBS, unlimited disk",
                    improves=[Bottleneck.disk_capacity, Bottleneck.memory],
                    degrades=[Bottleneck.disk_iops],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="Cluster too large",
            bottlenecks=[Bottleneck.disk_capacity],
        )
        alts = graph.suggest_alternatives(excuse)
        assert len(alts) == 2
        assert {a.to_family for a in alts} == {"i3en", "r7a"}

    def test_suggest_alternatives_no_bottleneck(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="i4i",
                    to_family="i3en",
                    trade_off="x",
                    improves=[Bottleneck.disk_capacity],
                ),
            ],
        )
        excuse = Excuse(instance="i4i.2xlarge", drive="gp3", reason="test")
        assert graph.suggest_alternatives(excuse) == []

    def test_suggest_alternatives_no_matching_edges(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="m6id",
                    to_family="m7a",
                    trade_off="x",
                    improves=[Bottleneck.disk_capacity],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="test",
            bottlenecks=[Bottleneck.disk_capacity],
        )
        assert graph.suggest_alternatives(excuse) == []

    def test_empty_graph(self):
        graph = FamilyGraph()
        assert not graph.traits
        assert not graph.edges


@pytest.mark.parametrize(
    "excuse_inst, current_inst, expected_tags",
    [
        ("r6a.2xlarge", "r6a.2xlarge", [ExcuseTag.current_shape]),
        ("r6a.4xlarge", "r6a.2xlarge", [ExcuseTag.same_family, ExcuseTag.size_up]),
        ("r6a.xlarge", "r6a.2xlarge", [ExcuseTag.same_family, ExcuseTag.size_down]),
        ("i4i.2xlarge", "r6a.2xlarge", [ExcuseTag.different_family]),
        ("r6a.2xlarge", None, []),
    ],
)
def test_compute_excuse_tags(excuse_inst, current_inst, expected_tags):
    assert compute_excuse_tags(excuse_inst, current_inst) == expected_tags


class TestFamilyGraphBuild:
    """Test FamilyGraph.build() construction from hardware data."""

    def test_no_excuses_uses_preferred_families(self):
        """FamilyGraph.build() with empty excuses uses preferred_families as base."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
        )
        assert len(graph.traits) > 0, (
            "Graph should be populated from preferred_families even with no excuses"
        )
        assert "i4i" in graph.traits
        assert "r6a" in graph.traits

    def test_none_preferred_gives_empty_graph(self):
        """preferred_families=None → empty base, no families imposed on the model."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=None,
        )
        assert len(graph.traits) == 0
        assert len(graph.edges) == 0

    def test_m_times_n_edges(self):
        """Graph has exactly n*(n-1) directed edges for n families."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
        )
        n = len(graph.traits)
        assert len(graph.edges) == n * (n - 1)

    def test_edges_use_bottleneck_enum(self):
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
        )
        for edge in graph.edges:
            for b in edge.improves + edge.degrades:
                assert isinstance(b, Bottleneck)


_FREQ_WORKLOAD = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1000),
        estimated_write_per_second=certain_int(1000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(100),
    ),
)


class TestExcuseFrequency:
    """Test the new Excuse.frequency aggregation behavior."""

    def test_default_frequency_excluded_from_dump(self):
        excuse = Excuse(instance="x", drive="y", reason="z")
        data = excuse.model_dump()
        assert "frequency" not in data

    def test_dedup_singleton_keeps_unset_frequency(self):
        original = Excuse(instance="x", drive="y", reason="z")
        result = deduplicate_excuses([original])
        assert len(result) == 1
        assert "frequency" not in result[0].model_dump()

    def test_dedup_sums_frequencies_for_same_key(self):
        excuses = [Excuse(instance="x", drive="y", reason="z") for _ in range(3)]
        result = deduplicate_excuses(excuses)
        assert len(result) == 1
        only = result[0]
        assert only.frequency == 3
        assert "frequency" in only.model_dump()

    def test_dedup_splits_on_bottleneck_difference(self):
        excuses = [
            Excuse(
                instance="x",
                drive="y",
                reason="z",
                bottlenecks=[Bottleneck.cpu],
            ),
            Excuse(
                instance="x",
                drive="y",
                reason="z",
                bottlenecks=[Bottleneck.memory],
            ),
        ]
        result = deduplicate_excuses(excuses)
        assert len(result) == 2

    def test_dedup_splits_on_tags_set_not_order(self):
        order_a = Excuse(
            instance="x",
            drive="y",
            reason="z",
            tags=[ExcuseTag.same_family, ExcuseTag.size_up],
        )
        order_b = Excuse(
            instance="x",
            drive="y",
            reason="z",
            tags=[ExcuseTag.size_up, ExcuseTag.same_family],
        )
        collapsed = deduplicate_excuses([order_a, order_b])
        assert len(collapsed) == 1
        assert collapsed[0].frequency == 2

        different_set = Excuse(
            instance="x",
            drive="y",
            reason="z",
            tags=[ExcuseTag.different_family],
        )
        split = deduplicate_excuses([order_a, different_set])
        assert len(split) == 2

    def test_plan_certain_explained_frequency_stays_unset(self):
        explained = planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_FREQ_WORKLOAD,
            extra_model_arguments={"require_local_disks": False},
        )
        assert len(explained.excuses) > 0
        for excuse in explained.excuses:
            assert "frequency" not in excuse.model_dump()

    def test_plan_explain_frequency_le_simulations(self):
        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_FREQ_WORKLOAD,
            simulations=4,
            extra_model_arguments={"require_local_disks": False},
        )
        excuses_flat = [
            e
            for node in walk_explanations(result.explanation.root)
            for e in node.excuses
        ]
        assert len(excuses_flat) > 0
        for excuse in excuses_flat:
            assert excuse.frequency <= 4


class TestExcuseBottlenecks:
    """Test the new plural `bottlenecks: List[Bottleneck]` field."""

    def test_default_bottlenecks_excluded_from_dump(self):
        excuse = Excuse(instance="x", drive="y", reason="z")
        data = excuse.model_dump()
        assert "bottlenecks" not in data

    def test_dump_uses_plural_not_singular(self):
        excuse = Excuse(
            instance="x",
            drive="y",
            reason="z",
            bottlenecks=[Bottleneck.cpu],
        )
        data = excuse.model_dump()
        assert data["bottlenecks"] == ["cpu"]
        assert "bottleneck" not in data

    def test_dedup_distinguishes_different_bottleneck_lists(self):
        excuses = [
            Excuse(
                instance="x",
                drive="y",
                reason="z",
                bottlenecks=[Bottleneck.cpu],
            ),
            Excuse(
                instance="x",
                drive="y",
                reason="z",
                bottlenecks=[Bottleneck.memory],
            ),
        ]
        result = deduplicate_excuses(excuses)
        assert len(result) == 2

    def test_dedup_distinguishes_same_set_different_order(self):
        excuses = [
            Excuse(
                instance="x",
                drive="y",
                reason="z",
                bottlenecks=[Bottleneck.cpu, Bottleneck.memory],
            ),
            Excuse(
                instance="x",
                drive="y",
                reason="z",
                bottlenecks=[Bottleneck.memory, Bottleneck.cpu],
            ),
        ]
        result = deduplicate_excuses(excuses)
        assert len(result) == 2

    def test_suggest_alternatives_matches_any_bottleneck(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="i4i",
                    to_family="i3en",
                    improves=[Bottleneck.disk_capacity],
                ),
                FamilyEdge(
                    from_family="i4i",
                    to_family="r7a",
                    improves=[Bottleneck.memory],
                ),
                FamilyEdge(
                    from_family="i4i",
                    to_family="c7a",
                    improves=[Bottleneck.cost],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="multi",
            bottlenecks=[Bottleneck.disk_capacity, Bottleneck.memory],
        )
        alts = graph.suggest_alternatives(excuse)
        assert {a.to_family for a in alts} == {"i3en", "r7a"}

    def test_suggest_alternatives_empty_when_no_bottlenecks(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="i4i",
                    to_family="i3en",
                    improves=[Bottleneck.disk_capacity],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="test",
            bottlenecks=[],
        )
        assert graph.suggest_alternatives(excuse) == []


_PREF_WORKLOAD = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1000),
        estimated_write_per_second=certain_int(1000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(100),
    ),
)


class TestPreferredFamilies:
    """PreferredFamilies.resolve() semantics."""

    def test_resolve_returns_none_when_completely_unset(self):
        assert PreferredFamilies().resolve() is None

    def test_resolve_returns_preset_set(self):
        pf = PreferredFamilies(preset=FamilyPreset.stateful_datastore)
        assert pf.resolve() == STATEFUL_DATASTORE_FAMILIES

        pf_stateless = PreferredFamilies(preset=FamilyPreset.stateless_service)
        assert pf_stateless.resolve() == STATELESS_SERVICE_FAMILIES

    def test_resolve_combines_preset_and_add(self):
        pf = PreferredFamilies(
            preset=FamilyPreset.stateful_datastore,
            add=frozenset({"i7i"}),
        )
        resolved = pf.resolve()
        assert resolved is not None
        assert "i7i" in resolved
        assert STATEFUL_DATASTORE_FAMILIES <= resolved

    def test_resolve_removes_from_preset(self):
        pf = PreferredFamilies(
            preset=FamilyPreset.stateful_datastore,
            remove=frozenset({"r5d"}),
        )
        resolved = pf.resolve()
        assert resolved is not None
        assert "r5d" not in resolved
        assert resolved == STATEFUL_DATASTORE_FAMILIES - {"r5d"}

    def test_resolve_no_preset_with_add_only(self):
        pf = PreferredFamilies(add=frozenset({"r7a", "m7a"}))
        assert pf.resolve() == frozenset({"r7a", "m7a"})

    def test_resolve_remove_only_returns_empty_set(self):
        pf = PreferredFamilies(remove=frozenset({"r5d"}))
        resolved = pf.resolve()
        assert resolved == frozenset()
        # Concrete empty frozenset, not None — "no preference" stays None.
        assert resolved is not None

    def test_resolve_remove_takes_precedence_over_add(self):
        pf = PreferredFamilies(
            add=frozenset({"i7i", "m7a"}),
            remove=frozenset({"i7i"}),
        )
        assert pf.resolve() == frozenset({"m7a"})

    def test_resolve_excludes_unset_in_dump(self):
        pf = PreferredFamilies(preset=FamilyPreset.stateful_datastore)
        data = pf.model_dump()
        assert "preset" in data
        assert "add" not in data
        assert "remove" not in data

        pf_full = PreferredFamilies(
            preset=FamilyPreset.stateful_datastore,
            add=frozenset({"i7i"}),
            remove=frozenset({"r5d"}),
        )
        data_full = pf_full.model_dump()
        assert "preset" in data_full
        assert "add" in data_full
        assert "remove" in data_full


class TestPlannerPreferredFamiliesOverride:
    """extra_model_arguments['preferred_families_override'] takes precedence."""

    def test_override_takes_precedence_over_model(self):
        """Cassandra default = STATEFUL_DATASTORE_FAMILIES (has r5d, no m5d).
        Override that adds m5d and removes r5d must be reflected in graph
        traits. m5d is used in place of i7i (not in loaded shapes catalog).
        """
        explained = planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_PREF_WORKLOAD,
            extra_model_arguments={
                "require_local_disks": False,
                "preferred_families_override": PreferredFamilies(
                    preset=FamilyPreset.stateful_datastore,
                    add=frozenset({"m5d"}),
                    remove=frozenset({"r5d"}),
                ),
            },
        )
        traits = explained.family_graph.traits
        assert "m5d" in traits
        assert "r5d" not in traits

    def test_override_resolves_to_frozenset_in_family_graph(self):
        """Override that pins a small explicit set: graph traits must be a
        subset of that set (current_shape_families is empty here)."""
        explicit = frozenset({"r7a", "m7a", "c7a"})
        explained = planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_PREF_WORKLOAD,
            extra_model_arguments={
                "require_local_disks": False,
                "preferred_families_override": PreferredFamilies(add=explicit),
            },
        )
        assert set(explained.family_graph.traits) <= explicit
        assert "i4i" not in explained.family_graph.traits

    def test_override_with_unset_preferred_families_returns_none(self):
        """A fully-empty PreferredFamilies override resolves to None and the
        planner falls through to model.preferred_families()."""
        explained = planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_PREF_WORKLOAD,
            extra_model_arguments={
                "require_local_disks": False,
                "preferred_families_override": PreferredFamilies(),
            },
        )
        traits = set(explained.family_graph.traits)
        # Cassandra's default preference is STATEFUL_DATASTORE_FAMILIES, so
        # the graph must come from that set (no current-cluster families
        # since desires has none).
        assert traits, "expected cassandra default preference to populate graph"
        assert traits <= STATEFUL_DATASTORE_FAMILIES


class _TestPreferredFamiliesModel(CapacityModel):
    """Minimal CapacityModel for backward-compat testing.

    Returns a PreferredFamilies (not FrozenSet) from preferred_families().
    capacity_plan defaults to None for every instance, so the planner
    produces zero plans and zero excuses — we only care about the
    family_graph derived from preferred_families().
    """

    @staticmethod
    def preferred_families():
        return PreferredFamilies(
            preset=FamilyPreset.stateful_datastore,
            remove=frozenset({"r5d"}),
        )

    @staticmethod
    def default_desires(user_desires, extra_model_arguments):
        # Bypass parent's access-pattern validation; tests pass throughput-ish
        # desires that we don't need to reshape.
        _ = extra_model_arguments
        return user_desires


@pytest.fixture
def _registered_pref_model():
    name = "test.preferred_families_model"
    planner.register_model(name, _TestPreferredFamiliesModel())
    try:
        yield name
    finally:
        planner._models.pop(name, None)  # pylint: disable=protected-access


class TestBackwardsCompatPreferredFamilies:
    """Models returning FrozenSet[str] still work without migration."""

    def test_model_returning_frozenset_still_works(self):
        """Cassandra returns a FrozenSet[str] — graph must build and
        contain the expected stateful-datastore families."""
        explained = planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_PREF_WORKLOAD,
            extra_model_arguments={"require_local_disks": False},
        )
        traits = set(explained.family_graph.traits)
        assert traits, "graph should be populated from cassandra's FrozenSet"
        assert traits <= STATEFUL_DATASTORE_FAMILIES
        assert "i4i" in traits
        assert "r6a" in traits

    def test_model_returning_preferred_families_resolved(self, _registered_pref_model):
        """A model returning PreferredFamilies(preset=..., remove={'r5d'})
        must produce a graph that omits r5d but includes other stateful
        families."""
        explained = planner.plan_certain_explained(
            model_name=_registered_pref_model,
            region="us-east-1",
            desires=_PREF_WORKLOAD,
        )
        traits = set(explained.family_graph.traits)
        assert "r5d" not in traits
        assert traits <= STATEFUL_DATASTORE_FAMILIES - {"r5d"}
        assert "i4i" in traits
        assert "r6a" in traits


class TestFamilyTraitEdgeContext:
    """Trait/Edge context slots are optional and clean by default."""

    def test_trait_context_default_excluded_from_dump(self):
        hardware = shapes.region("us-east-1")
        r6a_inst = next(
            inst for inst in hardware.instances.values() if inst.family == "r6a"
        )
        trait = FamilyTrait.from_instance(r6a_inst)
        data = trait.model_dump()
        assert "context" not in data

    def test_edge_context_default_excluded_from_dump(self):
        edge = FamilyEdge(from_family="r6a", to_family="r7a")
        data = edge.model_dump()
        assert "context" not in data

    def test_trait_with_context_round_trips(self):
        hardware = shapes.region("us-east-1")
        r6a_inst = next(
            inst for inst in hardware.instances.values() if inst.family == "r6a"
        )
        base = FamilyTrait.from_instance(r6a_inst)
        trait = base.model_copy(update={"context": {"flag": True, "n": 3}})
        data = trait.model_dump()
        assert data["context"] == {"flag": True, "n": 3}
        round_tripped = FamilyTrait.model_validate(data)
        assert round_tripped.context == {"flag": True, "n": 3}

    def test_edge_with_context_round_trips(self):
        edge = FamilyEdge(
            from_family="r6a",
            to_family="r7a",
            context={"note": "x", "score": 1.5},
        )
        data = edge.model_dump()
        assert data["context"] == {"note": "x", "score": 1.5}
        round_tripped = FamilyEdge.model_validate(data)
        assert round_tripped.context == {"note": "x", "score": 1.5}


class TestFamilyGraphHooks:
    """FamilyGraph.build honors derive_traits / derive_edge hooks."""

    def test_no_hooks_matches_current_build_shape(self):
        """Build with no hooks — output is identical to today's shape:
        every trait has context unset, every edge has context unset, and
        the existing test_no_excuses_uses_preferred_families /
        test_m_times_n_edges invariants still hold."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
        )
        assert "i4i" in graph.traits
        assert "r6a" in graph.traits
        n = len(graph.traits)
        assert len(graph.edges) == n * (n - 1)
        for trait in graph.traits.values():
            assert "context" not in trait.model_dump()
        for edge in graph.edges:
            assert "context" not in edge.model_dump()

    def test_derive_traits_populates_trait_context(self):
        hardware = shapes.region("us-east-1")
        target = "r6a"

        def hook(family: str, instance):
            _ = instance
            return {"flag": True} if family == target else {}

        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
            derive_traits=hook,
        )
        assert graph.traits[target].context == {"flag": True}
        for fam, trait in graph.traits.items():
            if fam == target:
                continue
            assert "context" not in trait.model_dump()

    def test_derive_edge_can_augment_improves(self):
        hardware = shapes.region("us-east-1")

        def hook(from_trait, to_trait, default_improves, default_degrades):
            _ = (from_trait, to_trait)
            return default_improves + [Bottleneck.cost], default_degrades, {}

        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
            derive_edge=hook,
        )
        assert graph.edges
        for edge in graph.edges:
            assert Bottleneck.cost in edge.improves

    def test_derive_edge_can_override_degrades(self):
        hardware = shapes.region("us-east-1")

        def hook(from_trait, to_trait, default_improves, default_degrades):
            _ = (from_trait, to_trait, default_improves, default_degrades)
            return [], [], {}

        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
            derive_edge=hook,
        )
        assert graph.edges
        for edge in graph.edges:
            assert edge.improves == []
            assert edge.degrades == []

    def test_derive_edge_context_excluded_when_empty(self):
        hardware = shapes.region("us-east-1")

        def hook(from_trait, to_trait, default_improves, default_degrades):
            _ = (from_trait, to_trait, default_improves, default_degrades)
            return [], [], {}

        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
            derive_edge=hook,
        )
        assert graph.edges
        for edge in graph.edges:
            assert "context" not in edge.model_dump()

    def test_derive_edge_context_set_when_nonempty(self):
        hardware = shapes.region("us-east-1")

        def hook(from_trait, to_trait, default_improves, default_degrades):
            _ = (from_trait, to_trait, default_improves, default_degrades)
            return [], [], {"note": "x"}

        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
            derive_edge=hook,
        )
        assert graph.edges
        for edge in graph.edges:
            assert edge.model_dump()["context"] == {"note": "x"}


class TestPublicDeriveEdgeAttributes:
    """The promoted `derive_edge_attributes` function still works."""

    def test_public_function_callable(self):
        hardware = shapes.region("us-east-1")
        r6a_inst = next(
            inst for inst in hardware.instances.values() if inst.family == "r6a"
        )
        r7a_inst = next(
            inst for inst in hardware.instances.values() if inst.family == "r7a"
        )
        trait_a = FamilyTrait.from_instance(r6a_inst)
        trait_b = FamilyTrait.from_instance(r7a_inst)
        result = derive_edge_attributes(trait_a, trait_b)
        assert isinstance(result, tuple)
        assert len(result) == 2
        improves, degrades = result
        assert isinstance(improves, list)
        assert isinstance(degrades, list)
        for b in improves + degrades:
            assert isinstance(b, Bottleneck)


class _TypedExplanation(ModelExplanation):
    """Test-only ModelExplanation subclass with typed fields."""

    rf_choice_reason: str
    page_cache_gib: float


class TestModelExplanationModel:
    """Base ModelExplanation behavior."""

    def test_default_context_excluded_from_dump(self):
        """ModelExplanation(model_name='x').model_dump() omits 'context'."""
        explanation = ModelExplanation(model_name="x")
        data = explanation.model_dump()
        assert data == {"model_name": "x"}
        assert "context" not in data

    def test_subclass_fields_appear_in_dump(self):
        """Subclass with typed fields surfaces them when set."""
        explanation = _TypedExplanation(
            model_name="test.typed",
            rf_choice_reason="three zones available",
            page_cache_gib=12.5,
        )
        data = explanation.model_dump()
        assert data["model_name"] == "test.typed"
        assert data["rf_choice_reason"] == "three zones available"
        assert data["page_cache_gib"] == 12.5

    def test_subclass_fields_excluded_when_unset(self):
        """Subclass fields are absent from model_dump() when unset."""

        # Required fields on the subclass mean we'd need to set them; use a
        # subclass where every added field has a default so we can test the
        # exclude-unset behavior cleanly.
        class _OptionalFields(ModelExplanation):
            note: str = ""
            score: float = 0.0

        explanation = _OptionalFields(model_name="test.optional")
        data = explanation.model_dump()
        assert data == {"model_name": "test.optional"}
        assert "note" not in data
        assert "score" not in data


class TestExplainedPlansModelExplanations:
    """ExplainedPlans.model_explanations behavior."""

    def test_default_explanations_excluded_from_dump(self):
        """ExplainedPlans(plans=[]).model_dump() omits 'model_explanations'."""
        explained = ExplainedPlans(plans=[])
        data = explained.model_dump()
        assert "model_explanations" not in data

    def test_explanations_round_trip_in_dump(self):
        """A populated model_explanations dict surfaces subclass fields."""
        explained = ExplainedPlans(
            plans=[],
            model_explanations={
                "test.typed": _TypedExplanation(
                    model_name="test.typed",
                    rf_choice_reason="three zones available",
                    page_cache_gib=12.5,
                ),
            },
        )
        data = explained.model_dump()
        assert "model_explanations" in data
        entry = data["model_explanations"]["test.typed"]
        assert entry["model_name"] == "test.typed"
        assert entry["rf_choice_reason"] == "three zones available"
        assert entry["page_cache_gib"] == 12.5


class TestPlannerExplainPlanHook:
    """plan_certain_explained calls model.explain_plan and aggregates.

    The hook only fires when a sub-model produced at least one plan, so we
    exercise it by monkey-patching an in-tree model that reliably produces
    plans (Cassandra) instead of registering a synthetic model that would
    have to re-implement `capacity_plan`. Patches are restored via
    try/finally — same teardown discipline as
    `TestBackwardsCompatPreferredFamilies`.
    """

    def test_default_no_explanations(self):
        """Cassandra's default explain_plan returns None — model_explanations
        must be empty and absent from model_dump."""
        explained = planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_PREF_WORKLOAD,
            extra_model_arguments={"require_local_disks": False},
        )
        assert not explained.model_explanations
        data = explained.model_dump()
        assert "model_explanations" not in data

    def test_model_with_explanation_populates_dict(self):
        """Patch Cassandra's explain_plan to return a typed
        ModelExplanation; verify it appears under the model's name and the
        subclass's typed fields round-trip through model_dump."""
        # pylint: disable=protected-access
        cassandra_model = planner._models["org.netflix.cassandra"]
        original = cassandra_model.explain_plan

        def _patched(plan, desires, extra_model_arguments):
            _ = (plan, desires, extra_model_arguments)
            return _TypedExplanation(
                model_name="org.netflix.cassandra",
                rf_choice_reason="three-zone-region",
                page_cache_gib=8.0,
            )

        cassandra_model.explain_plan = _patched
        try:
            explained = planner.plan_certain_explained(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=_PREF_WORKLOAD,
                extra_model_arguments={"require_local_disks": False},
            )
        finally:
            cassandra_model.explain_plan = original

        assert "org.netflix.cassandra" in explained.model_explanations
        entry = explained.model_explanations["org.netflix.cassandra"]
        assert isinstance(entry, _TypedExplanation)
        # Subclass-typed fields must round-trip through ExplainedPlans dump
        dumped = explained.model_dump()
        ce = dumped["model_explanations"]["org.netflix.cassandra"]
        assert ce["model_name"] == "org.netflix.cassandra"
        assert ce["rf_choice_reason"] == "three-zone-region"
        assert ce["page_cache_gib"] == 8.0

    def test_explanation_receives_chosen_plan(self):
        """The plan passed to explain_plan is sub_result.plans[0]: a real
        CapacityPlan with non-empty candidate_clusters."""
        # pylint: disable=protected-access,import-outside-toplevel
        from service_capacity_modeling.interface import CapacityPlan

        cassandra_model = planner._models["org.netflix.cassandra"]
        original = cassandra_model.explain_plan
        captured: dict = {}

        def _capturing(plan, desires, extra_model_arguments):
            _ = (desires, extra_model_arguments)
            captured["plan"] = plan
            return None

        cassandra_model.explain_plan = _capturing
        try:
            planner.plan_certain_explained(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=_PREF_WORKLOAD,
                extra_model_arguments={"require_local_disks": False},
            )
        finally:
            cassandra_model.explain_plan = original

        assert "plan" in captured, "explain_plan was not invoked"
        assert isinstance(captured["plan"], CapacityPlan)
        assert captured["plan"].candidate_clusters is not None
        # Non-empty candidate_clusters: at minimum a zonal or regional shape
        cc = captured["plan"].candidate_clusters
        assert cc.zonal or cc.regional


_LEAF_DESIRES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10),
        estimated_write_per_second=certain_int(10),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(1)),
)


def _node(name: str, **kwargs) -> ComposedExplanation:
    return ComposedExplanation(model_name=name, desires=_LEAF_DESIRES, **kwargs)


class TestComposedExplanationTree:
    """ComposedExplanation tree + walk_explanations utility."""

    def test_leaf_node_dump_excludes_defaults(self):
        data = _node("x").model_dump()
        assert "model_name" in data
        assert "desires" in data
        assert "excuses" not in data
        assert "regret_clusters" not in data
        assert "children" not in data

    def test_walk_explanations_yields_root_first(self):
        root = _node("only")
        walked = list(walk_explanations(root))
        assert len(walked) == 1
        assert walked[0] is root

    def test_walk_explanations_preorder_with_children(self):
        c1g1 = _node("c1g1")
        c2g1 = _node("c2g1")
        c1 = _node("c1", children=[c1g1])
        c2 = _node("c2", children=[c2g1])
        root = _node("root", children=[c1, c2])
        names = [n.model_name for n in walk_explanations(root)]
        assert names == ["root", "c1", "c1g1", "c2", "c2g1"]

    def test_walk_explanations_handles_none(self):
        assert not list(walk_explanations(None))


class TestPlanExplanationTree:
    """plan(..., explain=True) populates PlanExplanation.root."""

    _WORKLOAD = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1000),
            estimated_write_per_second=certain_int(1000),
            estimated_mean_read_latency_ms=certain_float(0.5),
            estimated_mean_write_latency_ms=certain_float(0.4),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(100)),
    )

    def test_root_is_populated(self):
        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=self._WORKLOAD,
            simulations=2,
            extra_model_arguments={"require_local_disks": False},
        )
        assert result.explanation.root is not None
        assert result.explanation.root.model_name == "org.netflix.cassandra"

    def test_excuses_appear_in_tree(self):
        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=self._WORKLOAD,
            simulations=2,
            extra_model_arguments={"require_local_disks": False},
        )
        excuses_flat = [
            e
            for node in walk_explanations(result.explanation.root)
            for e in node.excuses
        ]
        assert len(excuses_flat) > 0

    def test_cassandra_root_has_no_children(self):
        # Cassandra's CapacityModel does not currently override
        # compose_with, so the tree is leaf-only.
        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=self._WORKLOAD,
            simulations=2,
            extra_model_arguments={"require_local_disks": False},
        )
        assert result.explanation.root is not None
        assert result.explanation.root.children == []

    def test_plan_no_longer_accepts_explain_kwarg(self):
        """Layer G removes the legacy ``explain=`` parameter outright.

        Passing ``explain=True`` to ``plan`` must now raise ``TypeError``
        at the Python signature layer — this is the user-visible
        breaking change of Layer G.
        """
        with pytest.raises(TypeError):
            # pylint: disable=unexpected-keyword-arg
            planner.plan(  # type: ignore[call-arg]
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=self._WORKLOAD,
                simulations=1,
                explain=True,
                extra_model_arguments={"require_local_disks": False},
            )


class TestSampleRef:
    """SampleRef typed sample correlation."""

    def test_sample_ref_roundtrip(self):
        ref = SampleRef(sample_id="s-0001-deadbeef", sample_label="reads=1k")
        data = ref.model_dump()
        assert data == {"sample_id": "s-0001-deadbeef", "sample_label": "reads=1k"}
        round_tripped = SampleRef.model_validate(data)
        assert round_tripped == ref

    def test_sample_ref_for_desires_deterministic(self):
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1500),
                estimated_write_per_second=certain_int(500),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(250),
            ),
        )
        a = _sample_ref_for_desires(desires, 7)
        b = _sample_ref_for_desires(desires, 7)
        assert a == b
        assert a.sample_id.startswith("s-0007-")
        # Same digest length across runs.
        assert len(a.sample_id.split("-")[-1]) == 8

    def test_sample_ref_for_desires_label_includes_state_gib(self):
        desires = CapacityDesires(
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(2_000_000),
                estimated_write_per_second=certain_int(500),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(2048),
            ),
        )
        ref = _sample_ref_for_desires(desires, 0)
        assert "reads=2M" in ref.sample_label
        assert "writes=500" in ref.sample_label
        assert "state=2kGiB" in ref.sample_label

    def test_format_sample_value_scaling(self):
        assert _format_sample_value(500) == "500"
        assert _format_sample_value(1500) == "2k"
        assert _format_sample_value(1_500_000) == "2M"
        assert _format_sample_value(2_500_000_000) == "2B"
        assert _format_sample_value(100, suffix="GiB") == "100GiB"


class TestExcuseSampleAggregation:
    """deduplicate_excuses aggregates example_samples and sample_count."""

    def _excuse_with(self, *, instance="x", sample_id="s-0001-aaaa"):
        return Excuse(
            instance=instance,
            drive="y",
            reason="z",
            example_samples=[
                SampleRef(
                    sample_id=sample_id, sample_label="reads=1 writes=1 state=1GiB"
                )
            ],
        )

    def test_deduplicate_excuses_aggregates_example_samples(self):
        excuses = [
            self._excuse_with(sample_id="s-0001-a"),
            self._excuse_with(sample_id="s-0002-b"),
            self._excuse_with(sample_id="s-0003-c"),
        ]
        result = deduplicate_excuses(excuses)
        assert len(result) == 1
        only = result[0]
        assert only.frequency == 3
        assert only.sample_count == 3
        assert [s.sample_id for s in only.example_samples] == [
            "s-0001-a",
            "s-0002-b",
            "s-0003-c",
        ]

    def test_deduplicate_excuses_caps_example_samples_at_three(self):
        excuses = [self._excuse_with(sample_id=f"s-{i:04d}-{i:08x}") for i in range(7)]
        result = deduplicate_excuses(excuses)
        assert len(result) == 1
        only = result[0]
        assert len(only.example_samples) == MAX_EXAMPLE_SAMPLES
        assert only.sample_count == 7
        assert only.frequency == 7

    def test_deduplicate_excuses_collapses_duplicate_sample_ids(self):
        excuses = [
            self._excuse_with(sample_id="s-0001-a"),
            self._excuse_with(sample_id="s-0001-a"),
            self._excuse_with(sample_id="s-0002-b"),
        ]
        result = deduplicate_excuses(excuses)
        assert len(result) == 1
        only = result[0]
        assert only.frequency == 3
        assert only.sample_count == 2
        assert [s.sample_id for s in only.example_samples] == [
            "s-0001-a",
            "s-0002-b",
        ]

    def test_excuse_sample_count_distinct_from_frequency(self):
        # Three emissions, all from the same sample → frequency=3,
        # sample_count stays unset (default 1) because only one distinct
        # sample contributed.
        excuses = [self._excuse_with(sample_id="s-0001-a") for _ in range(3)]
        result = deduplicate_excuses(excuses)
        assert len(result) == 1
        only = result[0]
        assert only.frequency == 3
        dumped = only.model_dump()
        assert "sample_count" not in dumped
        assert dumped["example_samples"][0]["sample_id"] == "s-0001-a"

    def test_deduplicate_excuses_empty_sample_excuses_stay_clean(self):
        # No example_samples → both sample_count and example_samples
        # remain unset in JSON.
        excuses = [Excuse(instance="x", drive="y", reason="z") for _ in range(2)]
        result = deduplicate_excuses(excuses)
        assert len(result) == 1
        dumped = result[0].model_dump()
        assert "sample_count" not in dumped
        assert "example_samples" not in dumped


_EXPLAINED_WORKLOAD = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1000),
        estimated_write_per_second=certain_int(1000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(100)),
)


class TestPlanExplained:
    """planner.plan_explained returns ExplainedUncertainPlans."""

    # pylint: disable=import-outside-toplevel
    @staticmethod
    def _explained_imports():
        from service_capacity_modeling.explainability import ExplainedUncertainPlans
        from service_capacity_modeling.regret_explainability import RegretPlanSummary

        return ExplainedUncertainPlans, RegretPlanSummary

    def test_plan_explained_returns_explained_uncertain_plans(self):
        ExplainedUncertainPlans, _ = self._explained_imports()
        explained = planner.plan_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_EXPLAINED_WORKLOAD,
            simulations=2,
            extra_model_arguments={"require_local_disks": False},
        )
        assert isinstance(explained, ExplainedUncertainPlans)
        assert explained.plan.least_regret
        assert explained.family_graph.traits

    def test_plan_explained_summaries_one_per_least_regret(self):
        _, RegretPlanSummary = self._explained_imports()
        explained = planner.plan_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_EXPLAINED_WORKLOAD,
            simulations=2,
            extra_model_arguments={"require_local_disks": False},
        )
        assert len(explained.least_regret_summaries) == len(explained.plan.least_regret)
        for summary in explained.least_regret_summaries:
            assert isinstance(summary, RegretPlanSummary)

    def test_plan_explained_summary_missing_raises(self):
        # pylint: disable=import-outside-toplevel
        from service_capacity_modeling.capacity_planner import (
            _collect_composed_regret_candidates,
        )
        from service_capacity_modeling.regret_explainability import (
            summaries_for_least_regret,
        )

        explained = planner.plan_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_EXPLAINED_WORKLOAD,
            simulations=2,
            extra_model_arguments={"require_local_disks": False},
        )
        # Force a mismatch by handing summaries_for_least_regret a plan
        # whose topology never appeared as a candidate. The synthesized
        # plan reuses the real result's requirements but swaps the chosen
        # cluster for a different instance family to guarantee a fresh
        # topology hash.
        candidates = _collect_composed_regret_candidates(
            explained.plan.explanation.root
        )
        fake = explained.plan.least_regret[0].model_copy(deep=True)
        cluster = fake.candidate_clusters.zonal[0]
        renamed_instance = cluster.instance.model_copy(update={"name": "fake.0xlarge"})
        renamed_cluster = cluster.model_copy(update={"instance": renamed_instance})
        fake.candidate_clusters = fake.candidate_clusters.model_copy(
            update={"zonal": [renamed_cluster]}
        )
        with pytest.raises(KeyError):
            summaries_for_least_regret([fake], candidates)

    def test_plan_explained_considered_alternatives_capped(self):
        explained = planner.plan_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_EXPLAINED_WORKLOAD,
            simulations=4,
            extra_model_arguments={"require_local_disks": False},
            considered_alternatives_cap=3,
        )
        assert len(explained.considered_alternatives) <= 3
        # Selected topologies are not among considered_alternatives.
        # pylint: disable=import-outside-toplevel
        from service_capacity_modeling.regret_explainability import (
            topology_signature,
        )

        selected = {topology_signature(p) for p in explained.plan.least_regret}
        for summary in explained.considered_alternatives:
            assert topology_signature(summary.plan) not in selected

    def test_plan_explained_family_graph_populated(self):
        explained = planner.plan_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=_EXPLAINED_WORKLOAD,
            simulations=2,
            extra_model_arguments={"require_local_disks": False},
        )
        # Cassandra defaults to STATEFUL_DATASTORE_FAMILIES so the graph
        # is populated.
        traits = set(explained.family_graph.traits)
        assert traits
        assert traits <= STATEFUL_DATASTORE_FAMILIES

    def test_plan_explained_calls_explain_plan_hook_per_sub_model(self):
        # pylint: disable=protected-access
        cassandra_model = planner._models["org.netflix.cassandra"]
        original = cassandra_model.explain_plan
        captured: dict = {}

        def _capturing(plan, desires, extra_model_arguments):
            _ = (plan, desires, extra_model_arguments)
            captured.setdefault("count", 0)
            captured["count"] += 1
            return ModelExplanation(model_name="org.netflix.cassandra")

        cassandra_model.explain_plan = _capturing
        try:
            explained = planner.plan_explained(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=_EXPLAINED_WORKLOAD,
                simulations=2,
                extra_model_arguments={"require_local_disks": False},
            )
        finally:
            cassandra_model.explain_plan = original

        assert captured.get("count") == 1
        assert "org.netflix.cassandra" in explained.model_explanations
