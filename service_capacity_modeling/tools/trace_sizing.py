#!/usr/bin/env python3
# pylint: disable=too-many-statements
"""Trace why the model picks c5a.8xlarge instead of m7a.4xlarge."""

import math
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.models.common import (
    cpu_headroom_target,
    normalize_cores,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)
from service_capacity_modeling.models.utils import next_doubling


def main() -> None:
    buffers = NflxCassandraCapacityModel.default_buffers()
    current_instance = shapes.instance("m7a.4xlarge")
    current_count = 64
    current_cpu_util = 0.29
    current_total_cpu = float(current_instance.cpu * current_count)

    print("=" * 80)
    print("WHY THE MODEL REJECTS m7a.4xlarge x 64")
    print("=" * 80)

    candidates = [
        "m7a.4xlarge",  # Current instance (16 vCPU)
        "c5a.8xlarge",  # Model's EBS pick (32 vCPU)
        "c6id.4xlarge",  # Model's local pick (16 vCPU)
        "c6id.8xlarge",  # Model's local pick for current_clusters (32 vCPU)
    ]

    for inst_name in candidates:
        inst = shapes.instance(inst_name)
        headroom = cpu_headroom_target(inst, buffers)
        target_util = 1 - headroom

        # CPU calculation
        used_cpu = (current_cpu_util / target_util) * current_total_cpu
        used_cpu = math.ceil(used_cpu)

        # Normalize to target instance
        normalized = normalize_cores(
            core_count=used_cpu,
            target_shape=inst,
            reference_shape=current_instance,
        )

        # How many instances needed
        instances_for_cpu = math.ceil(normalized / inst.cpu)

        # With required_cluster_size=64, what's the actual cluster size?
        cluster_size = next_doubling(instances_for_cpu, base=64)

        # Total vCPU in that cluster
        total_vcpu = cluster_size * inst.cpu

        print(f"\n{'─' * 70}")
        print(f"{inst_name} ({inst.cpu} vCPU, ${inst.annual_cost:,.0f}/yr)")
        print(f"  headroom={headroom:.0%}, target_util={target_util:.0%}")
        print(f"  used_cpu = (0.29 / {target_util:.2f}) * 1024 = {used_cpu}")
        print(f"  normalized_cores = {normalized} (gen adjustment from m7a)")
        print(
            f"  instances_for_cpu = "
            f"ceil({normalized} / {inst.cpu}) "
            f"= {instances_for_cpu}"
        )
        print(
            f"  cluster_size = "
            f"next_doubling({instances_for_cpu}, "
            f"base=64) = {cluster_size}"
        )
        print(f"  total_vcpu = {cluster_size} * {inst.cpu} = {total_vcpu}")
        print(
            f"  cluster_annual_cost = "
            f"{cluster_size} * 3 zones "
            f"* ${inst.annual_cost:,.0f} "
            f"= ${cluster_size * 3 * inst.annual_cost:,.0f}"
        )

        if instances_for_cpu > 64:
            print(
                f"  >>> REJECTED: needs {instances_for_cpu} "
                f"instances but required_cluster_size=64"
            )
            print(f"  >>> Would need to DOUBLE to {cluster_size} nodes!")
        else:
            print(f"  >>> FITS: {instances_for_cpu} <= 64, cluster_size stays at 64")

    # The critical insight
    print(f"\n{'=' * 80}")
    print("THE ROOT CAUSE")
    print("=" * 80)
    print()
    print("CPU headroom breakdown for Cassandra:")
    print("  1. Background buffer:  ratio=2.0 (reserves 50% for compaction/repair)")
    print("  2. Compute buffer:     ratio=1.5 (general compute headroom)")
    print("  3. Combined effect:    target_util ≈ 28%")
    print()
    print("Current cluster reality:")
    print("  CPU utilization = 29% (ALREADY includes compaction running)")
    print("  29% > 28% target → model says cluster is OVER capacity")
    print("  → model wants MORE instances")
    print()
    print("With required_cluster_size=64:")
    print("  m7a.4xlarge: needs ceil(1061/16) = 67 > 64 → doubles to 128!")
    print("  c5a.8xlarge: needs ceil(1733/32) = 55 ≤ 64 → stays at 64")
    print()
    print("The model picks 8xlarge because 4xlarge would DOUBLE the cluster.")
    print("But the cluster runs fine at 29% on m7a.4xlarge x 64!")
    print()
    print("Core tension:")
    print("  The 2x background buffer assumes compaction isn't running yet.")
    print("  But in certain-mode with current_clusters, compaction IS running.")
    print("  The observed 29% CPU ALREADY INCLUDES background work.")
    print("  The buffer is double-counting.")


if __name__ == "__main__":
    main()
