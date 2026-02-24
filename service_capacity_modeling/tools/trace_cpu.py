#!/usr/bin/env python3
# pylint: disable=too-many-statements
"""Trace the CPU requirement calculation for the canonical example."""

import math
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    BufferComponent,
)
from service_capacity_modeling.models.common import (
    buffer_for_components,
    cpu_headroom_target,
    DerivedBuffers,
    normalize_cores,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)


def main() -> None:
    buffers = NflxCassandraCapacityModel.default_buffers()

    # Current cluster state
    current_instance = shapes.instance("m7a.4xlarge")
    current_count = 64
    current_cpu_util = 0.29  # 29%
    current_total_cpu = float(current_instance.cpu * current_count)

    print("=" * 80)
    print("CPU REQUIREMENT TRACE: cass_dgw_kv_graph_identity_nodes")
    print("=" * 80)
    print("\nCurrent cluster:")
    print(f"  Instance: m7a.4xlarge ({current_instance.cpu} vCPU)")
    print(f"  Count: {current_count}/zone")
    print(f"  CPU util: {current_cpu_util * 100:.0f}%")
    print(f"  Total vCPU: {current_total_cpu}")
    print(f"  Used vCPU: {current_cpu_util * current_total_cpu:.0f}")

    # What the Cassandra default buffers look like
    print("\nCassandra default buffers:")
    for name, buf in buffers.desired.items():
        print(f"  {name}: ratio={buf.ratio}, components={buf.components}")

    # Candidate instances to trace
    candidates = ["c5a.8xlarge", "c6a.8xlarge", "m6i.4xlarge", "m7a.4xlarge"]

    for inst_name in candidates:
        inst = shapes.instance(inst_name)
        print(f"\n{'─' * 80}")
        print(f"Candidate: {inst_name} ({inst.cpu} vCPU, {inst.ram_gib} GiB RAM)")
        print(f"{'─' * 80}")

        # Step 1: cpu_headroom_target
        headroom = cpu_headroom_target(inst, buffers)
        target_util = 1 - headroom
        print(f"  headroom = {headroom:.2%}")
        print(f"  target_util = 1 - headroom = {target_util:.2%}")

        # Step 2: used_cpu = (current_util / target_util) * current_total_cpu
        used_cpu = (current_cpu_util / target_util) * current_total_cpu
        print(
            f"  used_cpu = "
            f"({current_cpu_util:.2f} / {target_util:.2f})"
            f" * {current_total_cpu:.0f} = {used_cpu:.0f}"
        )

        # Step 3: derived_buffers
        derived_buffers = DerivedBuffers.for_components(
            buffers.derived, [BufferComponent.cpu]
        )
        print(
            f"  derived_buffers: "
            f"scale={derived_buffers.scale}, "
            f"preserve={derived_buffers.preserve}"
        )

        # Step 4: calculate_requirement
        final_cpu = derived_buffers.calculate_requirement(
            current_usage=used_cpu,
            existing_capacity=current_total_cpu,
        )
        final_cpu = math.ceil(final_cpu)
        print(f"  final_cpu (after derived_buffers) = {final_cpu}")

        # Step 5: normalize_cores
        reference_shape = current_instance
        normalized = normalize_cores(
            core_count=final_cpu,
            target_shape=inst,
            reference_shape=reference_shape,
        )
        print(
            f"  normalize_cores({final_cpu}, "
            f"target={inst_name}, "
            f"ref=m7a.4xlarge) = {normalized}"
        )

        # How many of these instances needed?
        instances_needed = math.ceil(normalized / inst.cpu)
        print(f"  Instances needed for CPU alone: {instances_needed}")

    # Also trace the Cassandra-specific background buffer effect
    print(f"\n{'=' * 80}")
    print("CASSANDRA BACKGROUND BUFFER ANALYSIS")
    print(f"{'=' * 80}")

    # The background buffer in Cassandra default_buffers():
    # Buffer(ratio=2.0, components=[cpu, network, BACKGROUND])
    cpu_buffer = buffer_for_components(buffers=buffers, components=["background"])
    print(f"\n  Background buffer ratio: {cpu_buffer.ratio}")
    print(f"  This means: {cpu_buffer.ratio}x multiplier on CPU for background tasks")
    print("  (compaction, backup, repair)")

    # The compute buffer:
    compute_buffer = buffer_for_components(
        buffers=buffers, components=[BufferComponent.compute]
    )
    print(f"\n  Compute buffer ratio: {compute_buffer.ratio}")

    cpu_only = buffer_for_components(buffers=buffers, components=[BufferComponent.cpu])
    print(f"\n  CPU buffer ratio (cpu + compute + background): {cpu_only.ratio}")

    # What if the cluster was right-sized for m7a.4xlarge?
    print(f"\n{'=' * 80}")
    print("RIGHT-SIZING ANALYSIS")
    print(f"{'=' * 80}")
    m7a = shapes.instance("m7a.4xlarge")
    headroom_m7a = cpu_headroom_target(m7a, buffers)
    target_util_m7a = 1 - headroom_m7a
    print(
        f"\n  m7a.4xlarge headroom: {headroom_m7a:.2%}, "
        f"target util: {target_util_m7a:.2%}"
    )
    print("  Current util: 29%")
    print(f"  Current is {'UNDER' if 0.29 < target_util_m7a else 'OVER'} the target")
    print(f"  Scale factor: {0.29 / target_util_m7a:.2f}x")
    print(
        f"  This means the current cluster uses "
        f"{0.29 / target_util_m7a * 100:.0f}% of target capacity"
    )


if __name__ == "__main__":
    main()
