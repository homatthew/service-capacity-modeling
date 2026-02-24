#!/usr/bin/env python3
# pylint: disable=too-many-locals,too-many-statements
"""Trace the disk density calculation.

Shows where the 4x buffer cancels (or doesn't).
"""

import math
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.models.common import (
    buffer_for_components,
    get_effective_disk_per_node_gib,
    DerivedBuffers,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)
from service_capacity_modeling.models.utils import next_doubling


def main() -> None:
    buffers = NflxCassandraCapacityModel.default_buffers()
    disk_buffer = buffer_for_components(
        buffers=buffers, components=[BufferComponent.disk]
    )
    print(f"Disk buffer ratio: {disk_buffer.ratio}")

    region = shapes.region("us-east-1")

    # MPL cluster: 675 GiB actual disk/node, 16 nodes/zone
    actual_disk_zone = 675.33 * 16  # 10,805 GiB
    volume_size = 1573
    allocated_disk_zone = volume_size * 16  # 25,168 GiB

    # What RequirementFromCurrentCapacity.disk_gib computes
    derived_disk = DerivedBuffers.for_components({}, [BufferComponent.disk])
    needed_disk = derived_disk.calculate_requirement(
        current_usage=actual_disk_zone,
        existing_capacity=allocated_disk_zone,
        desired_buffer_ratio=disk_buffer.ratio,
    )

    print("\nMPL cluster:")
    print(f"  Actual disk/zone: {actual_disk_zone:,.0f} GiB")
    print(f"  Allocated (vol*count): {allocated_disk_zone:,.0f} GiB")
    print(f"  needed_disk_gib (after buffers): {needed_disk:,.0f} GiB")
    print(f"  Ratio: {needed_disk / actual_disk_zone:.2f}x")

    # Now check density for EBS and local
    instances = [
        ("m7a.8xlarge", False),  # EBS candidate
        ("m6i.8xlarge", False),  # EBS candidate
        ("r6a.8xlarge", False),  # EBS candidate
        ("i4i.4xlarge", True),  # Local candidate
        ("i3en.3xlarge", True),  # Local candidate
        ("i3en.6xlarge", True),  # Local candidate
    ]

    for inst_name, is_local in instances:
        inst = shapes.instance(inst_name)
        drive = inst.drive if is_local else region.drives["gp3"]
        assert drive is not None, f"{inst_name} has no drive"

        disk_per_node = get_effective_disk_per_node_gib(
            inst,
            drive,
            disk_buffer.ratio,
            max_local_data_per_node_gib=1280,
            max_attached_data_per_node_gib=2048,
        )

        min_nodes = math.ceil(needed_disk / disk_per_node)
        cluster_size = next_doubling(min_nodes, base=16)

        storage_type = "local" if is_local else "EBS"
        physical_disk = inst.drive.size_gib if inst.drive else "N/A (EBS)"

        print(f"\n  {inst_name} ({storage_type}):")
        print(f"    Physical disk: {physical_disk}")
        print(f"    effective_disk_per_node: {disk_per_node:,.0f} GiB")
        br = disk_buffer.ratio
        if is_local:
            limit = 1280
            bl = limit * br
            assert inst.drive is not None
            sz = inst.drive.size_gib
            print(f"      = min({limit} * {br}, {sz}) = min({bl:.0f}, {sz})")
        else:
            limit = 2048
            bl = limit * br
            msz = drive.max_size_gib
            print(f"      = min({limit} * {br}, {msz}) = min({bl:.0f}, {msz})")
        print(
            f"    min_nodes = "
            f"ceil({needed_disk:,.0f} / "
            f"{disk_per_node:,.0f}) = {min_nodes}"
        )
        print(
            f"    cluster_size = next_doubling({min_nodes}, base=16) = {cluster_size}"
        )
        fits = cluster_size <= 16
        label = "FITS at 16" if fits else f"NEEDS {cluster_size} nodes"
        print(f"    >>> {label}")

        # Show the un-buffered math
        actual_per_node = actual_disk_zone / 16
        ok = "OK" if actual_per_node < limit else "OVER"
        print(
            f"    Reality: actual data/node = "
            f"{actual_per_node:.0f} GiB "
            f"vs limit {limit} GiB -> {ok}"
        )

    # Show WHY the cancellation doesn't work for EBS
    print(f"\n{'=' * 80}")
    print("WHY THE 4x BUFFER DOESN'T CANCEL FOR EBS")
    print(f"{'=' * 80}")

    ebs_limit = 2048
    local_limit = 1280
    gp3 = region.drives["gp3"]
    i4i = shapes.instance("i4i.4xlarge")
    br = disk_buffer.ratio

    print("\n  For EBS (max_attached_data=2048, gp3 max_size=16384):")
    ebs_effective = min(ebs_limit * br, gp3.max_size_gib)
    print(
        f"    effective = min({ebs_limit} * {br}, "
        f"{gp3.max_size_gib}) = {ebs_effective:.0f}"
    )
    ne_ratio = needed_disk / ebs_effective
    print(
        f"    needed/effective = "
        f"{needed_disk:,.0f} / {ebs_effective:,.0f} "
        f"= {ne_ratio:.1f}"
    )
    cancel_ratio = actual_disk_zone / ebs_limit
    print(
        f"    If buffer canceled: actual / limit = "
        f"{actual_disk_zone:,.0f} / {ebs_limit} "
        f"= {cancel_ratio:.1f}"
    )

    assert i4i.drive is not None
    phys = i4i.drive.size_gib
    print(f"\n  For i4i.4xlarge (max_local_data=1280, phys disk={phys}):")
    i4i_effective = min(local_limit * br, phys)
    print(f"    effective = min({local_limit} * {br}, {phys}) = {i4i_effective:.0f}")
    ni_ratio = needed_disk / i4i_effective
    print(
        f"    needed/effective = "
        f"{needed_disk:,.0f} / {i4i_effective:,.0f} "
        f"= {ni_ratio:.1f}"
    )
    lc_ratio = actual_disk_zone / local_limit
    print(
        f"    If buffer canceled: actual / limit = "
        f"{actual_disk_zone:,.0f} / {local_limit} "
        f"= {lc_ratio:.1f}"
    )

    print("\n  The 4x DOES cancel IF the effective limit = limit * 4.0")
    print("  But for local, physical disk is the BINDING constraint:")
    print("    min(1280 * 4, 3492) = 3492 <- physical disk wins, NOT limit * 4")
    print("  For EBS:")
    print("    min(2048 * 4, 16384) = 8192 <- limit * 4 wins (since 8192 < 16384)")
    ebs_r = 43221 / 8192
    ebs_c = math.ceil(ebs_r)
    print(
        f"    So EBS effective = 8192, and 43221/8192 = {ebs_r:.1f} -> ceil = {ebs_c}"
    )
    nb_r = 10805 / 2048
    nb_c = math.ceil(nb_r)
    print(f"    Without buffer: 10805/2048 = {nb_r:.1f} -> ceil = {nb_c}")
    print("    >>> Buffer DOES cancel for EBS (both give ~6 nodes)")
    print("\n  Wait... that means the density limit ISN'T the blocker for EBS?")
    print("  Let me recheck...")
    ebs_nodes = math.ceil(needed_disk / ebs_effective)
    print(
        f"  min_nodes_for_ebs = "
        f"ceil({needed_disk:.0f} / {ebs_effective:.0f}) "
        f"= {ebs_nodes}"
    )
    print(f"  That's only {ebs_nodes} nodes, which is <= 16. So disk density FITS!")

    print(f"\n{'=' * 80}")
    print("REVISED ANALYSIS: What actually blocks the EBS plan?")
    print(f"{'=' * 80}")
    print(f"  Disk density: FITS ({ebs_nodes} nodes needed)")
    print(
        "  The REAL blocker must be something else "
        "-- likely MEMORY or instance filtering"
    )


if __name__ == "__main__":
    main()
