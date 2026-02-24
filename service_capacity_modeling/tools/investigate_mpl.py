#!/usr/bin/env python3
# pylint: disable=broad-exception-caught,too-many-locals,too-many-statements
"""
Investigate cass_mpl_useast1 — read-heavy EBS cluster.

Current: m7a.8xlarge x 16/zone, gp3 1573 GiB
  - 120K reads/s, 3.3K writes/s (read-heavy)
  - 10.8 TiB data, no compression
  - 7.6% CPU utilization
  - 675 GiB disk used per node
"""

import math
from typing import Any

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    AccessPattern,
    BufferComponent,
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Drive,
    Interval,
    QueryPattern,
)
from service_capacity_modeling.models.common import (
    cpu_headroom_target,
    DerivedBuffers,
    normalize_cores,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)
from service_capacity_modeling.models.utils import next_doubling


def run_and_print(
    label: str, desires: CapacityDesires, extra_args: dict[str, Any]
) -> None:
    """Run planner and print detailed results."""
    print(f"\n{'─' * 100}")
    print(f"  {label}")
    print(f"{'─' * 100}")

    try:
        plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=3,
            extra_model_arguments=extra_args,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if not plans:
        print("  No plans generated")
        return

    for i, plan in enumerate(plans):
        cluster = plan.candidate_clusters.zonal[0]
        req = plan.requirements.zonal[0]
        ctx = req.context or {}
        params = cluster.cluster_params or {}

        print(f"\n  Plan #{i + 1} (rank={plan.rank}):")
        print(
            f"    Instance: {cluster.instance.name} x {cluster.count}/zone "
            f"({cluster.instance.cpu} cpu, {cluster.instance.ram_gib:.0f} GiB RAM)"
        )
        print(
            f"    Total annual cost: ${plan.candidate_clusters.total_annual_cost:,.0f}"
        )
        cc = plan.candidate_clusters
        zonal_cost = cc.annual_costs.get("cassandra.zonal-clusters", 0)
        print(f"    Cluster cost: ${zonal_cost:,.0f}")

        if cluster.attached_drives:
            d = cluster.attached_drives[0]
            print(
                f"    EBS: {d.name} {d.size_gib} GiB, "
                f"R:{d.read_io_per_s} W:{d.write_io_per_s} IOPS, "
                f"${d.annual_cost:,.0f}/vol/yr"
            )
        elif cluster.instance.drive:
            d = cluster.instance.drive
            print(f"    Local disk: {d.size_gib:,} GiB")

        print(f"    Heap: {params.get('cassandra.heap.gib', '?')} GiB")
        print(
            f"    Write buffer: {params.get('cassandra.heap.write.percent', '?')}, "
            f"Table: {params.get('cassandra.heap.table.percent', '?')}, "
            f"Compaction: {params.get('cassandra.compaction.min_threshold', '?')}"
        )
        print(
            f"    Working set: {ctx.get('working_set', 0) * 100:.1f}%, "
            f"RPS WS: {ctx.get('rps_working_set', 0) * 100:.1f}%, "
            f"Disk SLO WS: {ctx.get('disk_slo_working_set', 0) * 100:.1f}%"
        )
        print(
            f"    Req disk: {req.disk_gib.mid:.0f} GiB/zone, "
            f"Req mem: {req.mem_gib.mid:.0f} GiB/zone, "
            f"Req CPU: {req.cpu_cores.mid:.0f} cores/zone"
        )
        wb = ctx.get("write_buffer_gib", 0)
        print(f"    Write buffer needed: {wb:.1f} GiB/zone")
        print(f"    RF: {params.get('cassandra.keyspace.rf', '?')}")

        # Cost breakdown
        for cost_key, cost_val in sorted(plan.candidate_clusters.annual_costs.items()):
            print(f"    Cost [{cost_key}]: ${cost_val:,.0f}")


def main() -> None:
    print("=" * 100)
    print("INVESTIGATION: cass_mpl_useast1")
    print("  Currently: m7a.8xlarge x 16/zone, gp3 1573 GiB")
    print("  Reads: 120K/s, Writes: 3.3K/s, Data: 10.8 TiB (no compression)")
    print("  CPU: 7.6%, Disk/node: 675 GiB")
    print("=" * 100)

    current_cluster = CurrentZoneClusterCapacity(
        cluster_instance_name="m7a.8xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type="attached-ssd",
            size_gib=1573,
        ),
        cluster_instance_count=Interval(low=16, mid=16, high=16, confidence=1),
        cpu_utilization=Interval(low=2.98, mid=7.62, high=13.01, confidence=1),
        network_utilization_mbps=Interval(low=12.4, mid=41.6, high=282.0, confidence=1),
        disk_utilization_gib=Interval(
            low=663.75, mid=675.33, high=691.19, confidence=0.98
        ),
        cluster_type="cassandra",
    )

    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            estimated_read_per_second=Interval(
                low=26280, mid=120547, high=256665, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=226, mid=3284, high=11089, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=10620, mid=10805.3, high=11059, confidence=0.98
            ),
            estimated_compression_ratio=Interval(low=1, mid=1, high=1, confidence=1),
        ),
        current_clusters=CurrentClusters(zonal=[current_cluster] * 3),
    )

    extra_args_base = {"required_cluster_size": 16}

    # Scenario 1: EBS forced (what the model recommends for this EBS cluster)
    run_and_print(
        "Scenario 1: EBS forced (require_attached_disks=True)",
        desires,
        {
            **extra_args_base,
            "require_local_disks": False,
            "require_attached_disks": True,
        },
    )

    # Scenario 2: Local disks
    run_and_print(
        "Scenario 2: Local disks (require_local_disks=True)",
        desires,
        {**extra_args_base, "require_local_disks": True},
    )

    # Scenario 3: Auto
    run_and_print(
        "Scenario 3: Auto (require_local_disks=False)",
        desires,
        {**extra_args_base, "require_local_disks": False},
    )

    # ──────────────────────────────────────────────────────────
    # CPU Trace
    # ──────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("CPU REQUIREMENT TRACE")
    print("=" * 100)

    buffers = NflxCassandraCapacityModel.default_buffers()
    current_instance = shapes.instance("m7a.8xlarge")
    current_count = 16
    current_cpu_util = 0.0762
    current_total_cpu = float(current_instance.cpu * current_count)

    print("\nCurrent cluster:")
    cpu = current_instance.cpu
    ram = current_instance.ram_gib
    print(f"  Instance: m7a.8xlarge ({cpu} vCPU, {ram:.0f} GiB RAM)")
    print(f"  Count: {current_count}/zone")
    print(f"  CPU util: {current_cpu_util * 100:.1f}%")
    print(f"  Total vCPU: {current_total_cpu:.0f}")
    print(f"  Used vCPU: {current_cpu_util * current_total_cpu:.0f}")

    candidates = [
        "m7a.8xlarge",  # Current
        "m6i.8xlarge",  # Likely EBS pick
        "r6a.8xlarge",  # Memory-optimized
        "i3en.3xlarge",  # Storage-optimized local
        "i4i.4xlarge",  # Storage-optimized local
    ]

    for inst_name in candidates:
        inst = shapes.instance(inst_name)
        headroom = cpu_headroom_target(inst, buffers)
        target_util = 1 - headroom

        used_cpu = (current_cpu_util / target_util) * current_total_cpu

        derived_buffers = DerivedBuffers.for_components(
            buffers.derived, [BufferComponent.cpu]
        )
        final_cpu = math.ceil(
            derived_buffers.calculate_requirement(
                current_usage=used_cpu,
                existing_capacity=current_total_cpu,
            )
        )

        normalized = normalize_cores(
            core_count=final_cpu,
            target_shape=inst,
            reference_shape=current_instance,
        )
        instances_for_cpu = math.ceil(normalized / inst.cpu)
        cluster_size = next_doubling(instances_for_cpu, base=16)

        print(
            f"\n  {inst_name} ({inst.cpu} cpu, "
            f"{inst.ram_gib:.0f} GiB RAM, "
            f"${inst.annual_cost:,.0f}/yr):"
        )
        print(f"    headroom={headroom:.0%}, target_util={target_util:.0%}")
        print(
            f"    used_cpu = "
            f"({current_cpu_util:.4f} / {target_util:.2f})"
            f" * {current_total_cpu:.0f} = {used_cpu:.0f}"
        )
        print(f"    normalized = {normalized}")
        print(f"    instances_for_cpu = {instances_for_cpu}")
        print(
            f"    cluster_size = "
            f"next_doubling({instances_for_cpu}, "
            f"base=16) = {cluster_size}"
        )
        if instances_for_cpu <= 16:
            fits = "FITS"
        elif cluster_size == 32:
            fits = f"NEEDS {cluster_size} (would DOUBLE!)"
        else:
            fits = f"NEEDS {cluster_size} (would SCALE!)"
        print(f"    >>> {fits}")

    # ──────────────────────────────────────────────────────────
    # Disk/Memory Analysis
    # ──────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("DISK AND MEMORY ANALYSIS")
    print("=" * 100)

    disk_per_node = 675.33
    disk_total_zone = disk_per_node * 16
    print(f"\n  Disk per node: {disk_per_node:.0f} GiB")
    print(f"  Disk per zone: {disk_total_zone:,.0f} GiB")
    print(f"  EBS volume: 1573 GiB (utilization: {disk_per_node / 1573 * 100:.0f}%)")
    print("  Data (desires): 10,805 GiB")
    print(f"  Data per zone: {10805 / 3:.0f} GiB")
    ratio = disk_total_zone / (10805 / 3)
    print(f"  Disk vs data ratio: {ratio:.2f}x (RF=3 means ~3x = expected)")

    # Memory analysis: this is read-heavy, so working set matters
    print("\n  Memory analysis:")
    inst_ram = current_instance.ram_gib
    print(f"    m7a.8xlarge RAM: {inst_ram:.0f} GiB")
    heap_est = min(max(4, inst_ram // 2), 30)
    print(f"    Heap (estimated): {heap_est:.0f} GiB")
    avail_cache = inst_ram - heap_est - 4
    print(f"    Available for page cache: {avail_cache:.0f} GiB")
    print(f"    Data per node: {disk_per_node:.0f} GiB")
    working_set_pct = (current_instance.ram_gib - 30 - 4) / disk_per_node
    print(f"    Page cache as % of data: {working_set_pct * 100:.0f}%")


if __name__ == "__main__":
    main()
