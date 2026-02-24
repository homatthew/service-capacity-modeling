#!/usr/bin/env python3
# pylint: disable=too-many-positional-arguments
"""Trace the rps_working_set calculation for both clusters.

Shows why it doesn't cap.
"""

import math
from service_capacity_modeling.models.org.netflix.cassandra import _cass_io_per_read


def trace_rps_ws(
    label: str,
    reads_per_second_zone: int,
    needed_cores: int,
    instance_cpu: int,
    disk_used_gib_zone: float,
    zones: int,
    max_rps_to_disk: int = 500,
) -> None:
    estimated_cores_per_region = math.ceil((needed_cores * zones) / instance_cpu)
    instance_rps = max(1, reads_per_second_zone // estimated_cores_per_region)
    data_per_instance = max(
        1, (disk_used_gib_zone * zones) // estimated_cores_per_region
    )
    io_per_read = _cass_io_per_read(data_per_instance)
    disk_rps = instance_rps * io_per_read
    rps_working_set = min(1.0, disk_rps / max_rps_to_disk)

    print(f"\n  {label}:")
    print(f"    reads/zone: {reads_per_second_zone:,}")
    print(f"    needed_cores: {needed_cores}, instance_cpu: {instance_cpu}")
    print(f"    estimated_cores_per_region: {estimated_cores_per_region}")
    print(
        f"    instance_rps: "
        f"{reads_per_second_zone} // "
        f"{estimated_cores_per_region} = {instance_rps}"
    )
    print(f"    data_per_instance: {data_per_instance:,} GiB")
    print(f"    io_per_read (LCS levels): {io_per_read}")
    print(f"    disk_rps: {instance_rps} * {io_per_read} = {disk_rps}")
    print(
        f"    rps_working_set: "
        f"min(1.0, {disk_rps} / {max_rps_to_disk}) "
        f"= {rps_working_set}"
    )

    # What read rate would actually bring rps_working_set below 30.6%?
    target_ws = 0.306
    target_disk_rps = target_ws * max_rps_to_disk  # = 153
    target_instance_rps = target_disk_rps / io_per_read
    target_total_reads = target_instance_rps * estimated_cores_per_region
    print(f"    To get rps_ws < 30.6%: need disk_rps < {target_disk_rps:.0f}")
    print(f"      → instance reads < {target_instance_rps:.0f}/s")
    print(f"      → total reads/zone < {target_total_reads:.0f}/s")


print("=" * 80)
print("RPS WORKING SET TRACE")
print("=" * 80)

# MPL: 120K reads/s regional → 40K/zone, 3.3K writes
# Current capacity path uses observed CPU → needed_cores from current
# With 7.6% CPU, 32 vCPU, 16 nodes → ~39 used cores/zone
# But in _estimate_cassandra_requirement, needed_cores comes from
# RequirementFromCurrentCapacity.cpu() which gives much more
trace_rps_ws(
    "MPL (current_clusters path, EBS)",
    reads_per_second_zone=40182,  # 120547 / 3
    needed_cores=139,  # from the CPU trace (~139 for m6i.8xlarge target)
    instance_cpu=32,
    disk_used_gib_zone=10805,
    zones=3,
)

trace_rps_ws(
    "MPL (what if needed_cores were lower, like actual usage)",
    reads_per_second_zone=40182,
    needed_cores=39,  # actual used cores (7.6% of 512)
    instance_cpu=32,
    disk_used_gib_zone=10805,
    zones=3,
)

# Identity nodes: 101K reads/s regional → 33K/zone, 2.27M writes
trace_rps_ws(
    "Identity nodes (current_clusters path, EBS)",
    reads_per_second_zone=33846,  # 101538 / 3
    needed_cores=1061,  # from the CPU trace
    instance_cpu=32,
    disk_used_gib_zone=3591,  # 56 * 64
    zones=3,
)

# What about a write-heavy cluster with few reads?
trace_rps_ws(
    "Hypothetical write-heavy: 1K reads/s, 500K writes/s",
    reads_per_second_zone=333,  # 1000 / 3
    needed_cores=100,
    instance_cpu=32,
    disk_used_gib_zone=1000,
    zones=3,
)

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("""
The rps_working_set cap DOES work for low-read workloads, but:

1. It divides reads by estimated_cores_per_region (instance count proxy),
   giving per-instance reads. Even moderate read rates (40K/zone) spread
   across ~5 instances = 8K reads/instance.

2. Then it multiplies by io_per_read (LCS levels: 12-18 for large datasets).
   8K * 14 = 112K disk IOPS per instance — way above max_rps_to_disk=500.

3. So rps_working_set = min(1.0, 112000/500) = 1.0 — capped at 100%.

The formula assumes every read hits disk (no page cache!). It then says
"if all these reads hit disk, you'd need this much working set in memory."

For a cluster that ALREADY HAS page cache (like mpl with 88 GiB / 675 GiB = 13%),
the actual disk hit rate is much lower than 100%. But the model doesn't know that.
""")
