#!/usr/bin/env python3
# pylint: disable=too-many-statements
"""
Explain exactly where the 30.6% working set number comes from.
Show every step of the computation.
"""

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    AccessPattern,
    CapacityDesires,
    DataShape,
    QueryPattern,
    certain_int,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)
from service_capacity_modeling.stats import dist_for_interval


def main() -> None:
    print("=" * 80)
    print("WHERE DOES 30.6% COME FROM?")
    print("=" * 80)

    # Step 1: Get the inputs
    region = shapes.region("us-east-1")
    gp3 = region.drives["gp3"]

    print("\nStep 1: The two distributions")
    print("─" * 60)
    print("\n  A) EBS gp3 read latency distribution:")
    print("     From manual_drives.json:")
    print(
        f"     low={gp3.read_io_latency_ms.low}ms, "
        f"mid={gp3.read_io_latency_ms.mid}ms, "
        f"high={gp3.read_io_latency_ms.high}ms, "
        f"max={gp3.read_io_latency_ms.maximum_value}ms"
    )

    # Get the default Cassandra read SLO for latency pattern
    dummy_desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            estimated_read_per_second=certain_int(1000),
            estimated_write_per_second=certain_int(1000),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(100),
        ),
    )
    defaults = NflxCassandraCapacityModel.default_desires(dummy_desires, {})
    read_slo = defaults.query_pattern.read_latency_slo_ms

    print("\n  B) Cassandra read latency SLO distribution:")
    print("     From default_desires() for latency access pattern:")
    print(
        f"     min={read_slo.minimum_value}ms, "
        f"low={read_slo.low}ms, "
        f"mid={read_slo.mid}ms, "
        f"high={read_slo.high}ms, "
        f"max={read_slo.maximum_value}ms"
    )

    # Step 2: Convert to scipy distributions
    print("\n\nStep 2: Convert to probability distributions")
    print("─" * 60)

    drive_dist = dist_for_interval(gp3.read_io_latency_ms)
    slo_dist = dist_for_interval(read_slo)

    print("\n  Both are converted to scipy log-normal distributions")
    print("  (via dist_for_interval which fits a log-normal to the interval)")

    # Step 3: The computation
    print("\n\nStep 3: The computation (WorkingSetEstimator.working_set_percent)")
    print("─" * 60)

    target_percentile = 0.95  # Cassandra uses 0.95

    # How fast is the drive at the target percentile?
    minimum_drive_latency = drive_dist.ppf(target_percentile)
    print(f"\n  target_percentile = {target_percentile} (Cassandra sets this)")
    print(f"\n  a) minimum_drive_latency = drive_dist.ppf({target_percentile})")
    print(f"     = {minimum_drive_latency:.3f} ms")
    pct = target_percentile * 100
    print(
        f"     Meaning: {pct:.0f}% of EBS reads "
        f"complete within {minimum_drive_latency:.1f}ms"
    )

    # How much of the SLO distribution falls below that latency?
    required_percent = float(slo_dist.cdf(minimum_drive_latency))
    print(f"\n  b) required_percent = slo_dist.cdf({minimum_drive_latency:.3f})")
    print(f"     = {required_percent:.4f}")
    print(f"     = {required_percent * 100:.1f}%")

    print(f"\n  RESULT: working_set = {required_percent * 100:.1f}%")

    # Step 4: Explain what this means
    print("\n\nStep 4: What does this mean?")
    print("─" * 60)
    print(f"""
  The logic asks:

  "If a read hits EBS, {target_percentile * 100:.0f}% of the time it takes
   <= {minimum_drive_latency:.1f}ms. What fraction of our read SLO
   budget is consumed by a disk read at that latency?"

  Answer: {required_percent * 100:.1f}% of the read SLO distribution
  falls at or below {minimum_drive_latency:.1f}ms.

  The MODEL INTERPRETS this as:
  "{required_percent * 100:.1f}% of reads CANNOT AFFORD to hit disk,
   because the disk latency alone would consume their entire
   latency budget. Therefore, {required_percent * 100:.1f}% of data
   must be served from memory."

  Put differently: if your SLO says "P50 = 2ms, P95 = 5ms",
  and EBS P95 = {minimum_drive_latency:.1f}ms, then ~{required_percent * 100:.0f}% of
  your read latency budget is already consumed by disk alone.
  Those reads must come from memory instead.
""")

    # Step 5: Show how it varies
    print("\nStep 5: How the number changes with different inputs")
    print("─" * 60)

    # Vary the SLO
    print("\n  Varying the read SLO (keeping EBS gp3 drive):")
    slos = [
        (
            "Tight: P50=1ms, P95=3ms",
            {
                "minimum_value": 0.2,
                "low": 0.4,
                "mid": 1,
                "high": 3,
                "maximum_value": 5,
                "confidence": 0.98,
            },
        ),
        (
            "Default Cass latency: P50=2ms, P95=5ms",
            {
                "minimum_value": 0.2,
                "low": 0.4,
                "mid": 2,
                "high": 5,
                "maximum_value": 10,
                "confidence": 0.98,
            },
        ),
        (
            "Relaxed: P50=5ms, P95=20ms",
            {
                "minimum_value": 1,
                "low": 2,
                "mid": 5,
                "high": 20,
                "maximum_value": 50,
                "confidence": 0.98,
            },
        ),
        (
            "Very relaxed: P50=10ms, P95=50ms",
            {
                "minimum_value": 2,
                "low": 5,
                "mid": 10,
                "high": 50,
                "maximum_value": 100,
                "confidence": 0.98,
            },
        ),
        (
            "Default Cass throughput: P50=8ms, P95=90ms",
            {
                "minimum_value": 1,
                "low": 2,
                "mid": 8,
                "high": 90,
                "maximum_value": 100,
                "confidence": 0.98,
            },
        ),
    ]

    for label, slo_params in slos:
        from service_capacity_modeling.interface import FixedInterval

        slo = FixedInterval(**slo_params)
        slo_d = dist_for_interval(slo)
        drive_p95 = drive_dist.ppf(0.95)
        ws = float(slo_d.cdf(drive_p95))
        print(f"    {label}")
        print(
            f"      drive P95 = {drive_p95:.1f}ms, "
            f"SLO CDF at that point = {ws * 100:.1f}%"
        )

    # Vary the drive
    print("\n  Varying the drive (keeping default Cass latency SLO):")
    drives_to_check = ["gp3", "gp2", "io2"]
    slo_d = dist_for_interval(read_slo)

    for drive_name in drives_to_check:
        d = region.drives[drive_name]
        d_dist = dist_for_interval(d.read_io_latency_ms)
        d_p95 = d_dist.ppf(0.95)
        ws = float(slo_d.cdf(d_p95))
        print(
            f"    {drive_name}: read latency mid={d.read_io_latency_ms.mid}ms, "
            f"P95={d_p95:.1f}ms → working_set = {ws * 100:.1f}%"
        )

    # Local NVMe (i4i)
    i4i = shapes.instance("i4i.4xlarge")
    if i4i.drive and i4i.drive.read_io_latency_ms:
        d_dist = dist_for_interval(i4i.drive.read_io_latency_ms)
        d_p95 = d_dist.ppf(0.95)
        ws = float(slo_d.cdf(d_p95))
        print(
            f"    i4i NVMe: read latency mid={i4i.drive.read_io_latency_ms.mid}ms, "
            f"P95={d_p95:.1f}ms → working_set = {ws * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
