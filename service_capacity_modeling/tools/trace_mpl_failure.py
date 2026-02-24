#!/usr/bin/env python3
# pylint: disable=too-many-locals,too-many-statements
"""Trace exactly WHY the EBS path fails for cass_mpl_useast1."""

import math
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.models.common import (
    buffer_for_components,
    DerivedBuffers,
    working_set_from_drive_and_slo,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
    _cass_heap,
)
from service_capacity_modeling.models.utils import next_doubling
from service_capacity_modeling.stats import dist_for_interval
from service_capacity_modeling.interface import (
    CapacityDesires,
    DataShape,
    Interval,
    QueryPattern,
    AccessPattern,
)


def main() -> None:
    print("=" * 100)
    print("WHY EBS PLANS FAIL FOR cass_mpl_useast1")
    print("=" * 100)

    buffers = NflxCassandraCapacityModel.default_buffers()
    region = shapes.region("us-east-1")

    # Parameters
    required_cluster_size = 16
    max_attached = 2048

    # Disk requirement from current clusters
    disk_per_node_used = 675.33
    disk_per_zone_used = disk_per_node_used * 16  # 10,805

    # The storage buffer (4x)
    disk_buffer = buffer_for_components(
        buffers=buffers, components=[BufferComponent.disk]
    )
    br = disk_buffer.ratio
    print(f"\nStorage buffer ratio: {br}")

    # What the model computes for disk
    derived_disk = DerivedBuffers.for_components({}, [BufferComponent.disk])
    req_disk = derived_disk.calculate_requirement(
        current_usage=disk_per_zone_used,
        existing_capacity=1573 * 16,
        desired_buffer_ratio=br,
    )
    print(f"Disk requirement: {req_disk:,.0f} GiB/zone")
    buffered = disk_per_zone_used * br
    print(f"  = {disk_per_zone_used:,.0f} * {br} = {buffered:,.0f}")

    per_node_disk = req_disk / required_cluster_size
    print(f"Per-node disk needed: {per_node_disk:,.0f} GiB")
    print(f"Max EBS data per node: {max_attached} GiB")
    exceeds = per_node_disk > max_attached
    print(f">>> {'EXCEEDS LIMIT' if exceeds else 'FITS'}")

    min_nodes = math.ceil(req_disk / max_attached)
    cluster_after = next_doubling(min_nodes, base=16)
    print(f"Min nodes for disk: ceil({req_disk:.0f} / {max_attached}) = {min_nodes}")
    print(f"After doubling: next_doubling({min_nodes}, base=16) = {cluster_after}")

    # FAILURE POINT 1: Disk density
    print(f"\n{'-' * 80}")
    print("FAILURE POINT 1: DISK DENSITY")
    print(f"{'-' * 80}")
    print(
        f"  4x storage buffer inflates "
        f"{disk_per_zone_used:,.0f} -> "
        f"{req_disk:,.0f} GiB/zone"
    )
    print(f"  At 16 nodes: {per_node_disk:,.0f} GiB/node > 2,048 GiB max")
    print("  Model needs 32 nodes, but required_cluster_size=16 -> PLAN REJECTED")
    print(
        f"  Reality: actual disk is "
        f"{disk_per_node_used:.0f} GiB/node "
        f"(43% of 1573 GiB volume)"
    )

    # FAILURE POINT 2: Memory / working set
    print(f"\n{'-' * 80}")
    print("FAILURE POINT 2: EBS WORKING SET MEMORY TAX")
    print(f"{'-' * 80}")

    # Compute working set for EBS drive
    gp3 = region.drives["gp3"]
    default_desires = NflxCassandraCapacityModel.default_desires(
        CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern(AccessPattern.latency),
                estimated_read_per_second=Interval(
                    low=26280,
                    mid=120547,
                    high=256665,
                    confidence=0.98,
                ),
                estimated_write_per_second=Interval(
                    low=226,
                    mid=3284,
                    high=11089,
                    confidence=0.98,
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=10620,
                    mid=10805,
                    high=11059,
                    confidence=0.98,
                ),
            ),
        ),
        {},
    )

    qp = default_desires.query_pattern
    ds = default_desires.data_shape
    ws = working_set_from_drive_and_slo(
        drive_read_latency_dist=dist_for_interval(gp3.read_io_latency_ms),
        read_slo_latency_dist=dist_for_interval(qp.read_latency_slo_ms),
        estimated_working_set=(ds.estimated_working_set_percent),
        target_percentile=0.95,
    ).mid

    print(f"  EBS gp3 working set: {ws * 100:.1f}%")
    needed_mem_zone = ws * disk_per_zone_used
    print(
        f"  Memory needed (working set): "
        f"{ws:.3f} * {disk_per_zone_used:,.0f} "
        f"= {needed_mem_zone:,.0f} GiB/zone"
    )
    per_node_mem = needed_mem_zone / required_cluster_size
    heap = _cass_heap(122)  # m7a.8xlarge RAM
    base_mem = 4  # reserved_instance_app_mem_gib
    total_per_node = per_node_mem + heap + base_mem
    avail_ram = 122  # m7a.8xlarge
    print(
        f"  Per node: {per_node_mem:.0f} GiB WS "
        f"+ {heap:.0f} heap + {base_mem} base "
        f"= {total_per_node:.0f} GiB"
    )
    print(f"  m7a.8xlarge RAM: {avail_ram} GiB")
    over = "EXCEEDS RAM" if total_per_node > avail_ram else "FITS"
    print(f"  >>> {over}")

    # What instance would be needed for memory alone?
    print(f"\n  Minimum RAM per node for EBS: {total_per_node:.0f} GiB")
    print(f"  At 16 nodes, need instances with >= {total_per_node:.0f} GiB RAM")
    for inst_name in [
        "r6a.8xlarge",
        "r6a.12xlarge",
        "r6a.16xlarge",
    ]:
        inst = shapes.instance(inst_name)
        inst_heap = _cass_heap(inst.ram_gib)
        avail = inst.ram_gib - inst_heap - base_mem
        ok = "OK" if avail >= per_node_mem else "TOO SMALL"
        print(
            f"    {inst_name}: "
            f"{inst.ram_gib:.0f} GiB "
            f"- {inst_heap:.0f} heap "
            f"- {base_mem} base "
            f"= {avail:.0f} GiB avail -> {ok}"
        )

    # For local NVMe, what's the working set?
    print("\n  For comparison, local NVMe working set:")
    i4i = shapes.instance("i4i.4xlarge")
    if i4i.drive:
        ws_local = working_set_from_drive_and_slo(
            drive_read_latency_dist=dist_for_interval(i4i.drive.read_io_latency_ms),
            read_slo_latency_dist=dist_for_interval(qp.read_latency_slo_ms),
            estimated_working_set=(ds.estimated_working_set_percent),
            target_percentile=0.95,
        ).mid
        needed_local = ws_local * disk_per_zone_used
        print(f"  Working set: {ws_local * 100:.1f}%")
        per_node_local = needed_local / 16
        print(
            f"  Memory needed: "
            f"{needed_local:,.0f} GiB/zone "
            f"= {per_node_local:.0f} GiB/node"
        )

    # REALITY CHECK
    print(f"\n{'-' * 80}")
    print("REALITY CHECK")
    print(f"{'-' * 80}")
    print("  The cluster runs fine on m7a.8xlarge x 16 with gp3 1573 GiB")
    print(f"  Actual disk/node: {disk_per_node_used:.0f} GiB (not {per_node_disk:.0f})")
    print("  Actual CPU: 7.6% (target: 29% -> well under)")
    vol_util = disk_per_node_used / 1573 * 100
    print(f"  Actual volume utilization: {vol_util:.0f}%")
    cache_pct = 88 / disk_per_node_used * 100
    print(f"  Available page cache: ~88 GiB/node = {cache_pct:.0f}% of data")
    print()
    print("  MODEL says: no valid EBS plan at 16 nodes")
    print("  REALITY: cluster is healthy with plenty of headroom")
    print()
    print("  Root causes:")
    print(
        f"    1. 4x storage buffer: "
        f"{disk_per_zone_used:,.0f} -> "
        f"{req_disk:,.0f} GiB/zone"
    )
    print(f"       ({per_node_disk:,.0f} GiB/node > 2,048 max)")
    print(f"    2. 30.6% EBS working set: {needed_mem_zone:,.0f} GiB memory/zone")
    print(f"       ({total_per_node:.0f} GiB/node > 122 GiB m7a.8xlarge RAM)")
    print("    3. required_cluster_size=16 prevents scaling out")


if __name__ == "__main__":
    main()
