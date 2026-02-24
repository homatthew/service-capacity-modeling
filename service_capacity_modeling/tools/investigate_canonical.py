#!/usr/bin/env python3
# pylint: disable=broad-exception-caught
"""
Investigate the canonical example: cass_dgw_kv_graph_identity_nodes

This cluster runs on m7a.4xlarge x 64/zone with gp3 269 GiB volumes.
The model recommends something more expensive. Why?

Key stats:
- 2.27M writes/s (regional), ~101K reads/s
- 3.6 TiB data, compression=1 (no compression)
- 29% CPU utilization on m7a.4xlarge
- 56 GiB disk used per node
- Required cluster size: 64
"""

from typing import Any

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Drive,
    Interval,
    QueryPattern,
    AccessPattern,
)


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
            f"({cluster.instance.cpu} cpu, {cluster.instance.ram_gib} GiB RAM)"
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


def main() -> None:
    print("=" * 100)
    print("CANONICAL EXAMPLE: cass_dgw_kv_graph_identity_nodes")
    print("  Currently: m7a.4xlarge x 64/zone, gp3 269 GiB")
    print("  Writes: 2.27M/s, Reads: 101K/s, Data: 3.6 TiB (no compression)")
    print("=" * 100)

    # Current cluster capacity (from the JSON)
    current_cluster = CurrentZoneClusterCapacity(
        cluster_instance_name="m7a.4xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type="attached-ssd",
            size_gib=269,
        ),
        cluster_instance_count=Interval(low=62, mid=64, high=64, confidence=1),
        cpu_utilization=Interval(low=15.1, mid=29.0, high=43.8, confidence=1),
        network_utilization_mbps=Interval(low=3.7, mid=16.0, high=39.1, confidence=1),
        disk_utilization_gib=Interval(low=55.8, mid=56.1, high=56.5, confidence=0.98),
        cluster_type="cassandra",
    )

    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            estimated_read_per_second=Interval(
                low=817, mid=101538, high=185022, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=1139235, mid=2267043, high=3366875, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=3572.2, mid=3591.5, high=3617.2, confidence=0.98
            ),
            estimated_compression_ratio=Interval(low=1, mid=1, high=1, confidence=1),
        ),
        current_clusters=CurrentClusters(zonal=[current_cluster] * 3),
    )

    extra_args_base = {
        "required_cluster_size": 64,
    }

    # Scenario 1: As-is (require_local_disks=True, the default)
    run_and_print(
        "Scenario 1: Default (require_local_disks=True)",
        desires,
        {**extra_args_base, "require_local_disks": True},
    )

    # Scenario 2: EBS forced
    run_and_print(
        "Scenario 2: EBS forced (require_attached_disks=True)",
        desires,
        {
            **extra_args_base,
            "require_local_disks": False,
            "require_attached_disks": True,
        },
    )

    # Scenario 3: Auto (model chooses)
    run_and_print(
        "Scenario 3: Auto (require_local_disks=False)",
        desires,
        {**extra_args_base, "require_local_disks": False},
    )

    # Scenario 4: Without current_clusters (fresh provisioning)
    desires_fresh = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            estimated_read_per_second=Interval(
                low=817, mid=101538, high=185022, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=1139235, mid=2267043, high=3366875, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=3572.2, mid=3591.5, high=3617.2, confidence=0.98
            ),
            estimated_compression_ratio=Interval(low=1, mid=1, high=1, confidence=1),
        ),
    )

    run_and_print(
        "Scenario 4: Fresh provisioning, EBS forced",
        desires_fresh,
        {"require_local_disks": False, "require_attached_disks": True},
    )

    run_and_print(
        "Scenario 5: Fresh provisioning, local disks",
        desires_fresh,
        {"require_local_disks": True},
    )

    run_and_print(
        "Scenario 6: Fresh provisioning, auto",
        desires_fresh,
        {"require_local_disks": False},
    )

    # Scenario 7: What does the model compute for write buffer?
    print("\n\n" + "=" * 100)
    print("WRITE BUFFER ANALYSIS")
    print("=" * 100)

    # Write buffer calculation:
    # write_bytes_per_second = 2,267,043 * default_write_size (256 bytes)
    # = 580,363,008 bytes/s
    # But with current clusters, it uses the _estimate from current
    # Let's calculate manually
    wps = 2_267_043
    # Default write size for latency pattern: mid=256 bytes
    write_size = 256
    wbps = wps * write_size
    print(f"  Write bytes/s (regional): {wbps:,} ({wbps / (1 << 30):.2f} GiB/s)")

    for min_threshold in [4, 8, 16]:
        compactions_per_hour = 2
        hour_in_seconds = 3600
        write_buffer_gib = (
            (wbps * hour_in_seconds) / (min_threshold**compactions_per_hour)
        ) / (1 << 30)
        per_zone = write_buffer_gib / 3
        print(
            f"  min_threshold={min_threshold}: "
            f"write_buffer={write_buffer_gib:.1f} GiB regional, "
            f"{per_zone:.1f} GiB/zone"
        )

    # What if we reference the actual current disk utilization?
    print("\n  Current cluster analysis:")
    print("    Disk per node: 56 GiB")
    print(f"    Total per zone: 56 * 64 = {56 * 64:,} GiB")
    print(f"    Total regional: {56 * 64 * 3:,} GiB")
    print("    Data size (desires): 3,591 GiB")
    print(f"    Ratio disk_used/data_size: {56 * 64 * 3 / 3591:.2f}x")

    # What the CPU requirement looks like
    print("\n  CPU analysis:")
    used = 16 * 0.29
    print(f"    Current: m7a.4xlarge = 16 vCPU, 29% util = {used:.1f} used cores/node")
    zone_cores = 64 * used
    print(
        f"    Per zone: 64 nodes * {used:.1f} = {zone_cores:.0f} effective cores/zone"
    )
    print(f"    Total regional CPU demand: {3 * 64 * 16 * 0.29:.0f} effective cores")


if __name__ == "__main__":
    main()
