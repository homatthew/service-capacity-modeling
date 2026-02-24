#!/usr/bin/env python3
# pylint: disable=broad-exception-caught
"""
EBS Disk Investigation: Run concrete scenarios through the Cassandra model.

Investigates two reported problems:
1. Small EBS volumes — new provisionings get tiny EBS disks (100-200 GiB)
2. EBS regret on write-heavy workloads — EBS adds latency, should have been local

Runs a matrix of scenarios comparing EBS vs local-disk recommendations.

Usage:
    python -m service_capacity_modeling.tools.ebs_investigation
"""

from typing import Any, Optional

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    certain_int,
    DataShape,
    QueryPattern,
)


def run_scenario(
    name: str,
    desires: CapacityDesires,
    mode: str = "ebs",
    extra_args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run a single scenario and extract key metrics.

    mode: "ebs" forces attached disks, "local" forces local disks,
          "auto" uses require_local_disks=False (model chooses freely)
    """
    args = extra_args or {}
    if mode == "ebs":
        args["require_local_disks"] = False
        args["require_attached_disks"] = True
    elif mode == "local":
        args["require_local_disks"] = True
        args["require_attached_disks"] = False
    else:  # auto
        args["require_local_disks"] = False
        args["require_attached_disks"] = False

    try:
        cap_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=1,
            extra_model_arguments=args,
        )
    except Exception as e:
        return {"scenario": name, "error": str(e)}

    if not cap_plans:
        return {"scenario": name, "error": "No capacity plans generated"}

    plan = cap_plans[0]
    cluster = plan.candidate_clusters.zonal[0]
    req = plan.requirements.zonal[0]

    result: dict[str, Any] = {
        "scenario": name,
        "storage_type": mode,
        "instance": cluster.instance.name,
        "family": cluster.instance.family,
        "cpu": cluster.instance.cpu,
        "ram_gib": cluster.instance.ram_gib,
        "count_per_zone": cluster.count,
        "total_zones": 3,
        "total_annual_cost": float(plan.candidate_clusters.total_annual_cost),
        "cluster_cost": float(
            plan.candidate_clusters.annual_costs.get("cassandra.zonal-clusters", 0)
        ),
    }

    # Cassandra params
    params = cluster.cluster_params or {}
    result["heap_gib"] = params.get("cassandra.heap.gib", "N/A")
    result["write_buffer_pct"] = params.get("cassandra.heap.write.percent", "N/A")
    result["table_buffer_pct"] = params.get("cassandra.heap.table.percent", "N/A")
    result["compaction_min_threshold"] = params.get(
        "cassandra.compaction.min_threshold", "N/A"
    )
    result["rf"] = params.get("cassandra.keyspace.rf", "N/A")

    # Requirement context
    ctx = req.context or {}
    result["working_set_pct"] = round(ctx.get("working_set", 0) * 100, 1)
    result["rps_working_set_pct"] = round(ctx.get("rps_working_set", 0) * 100, 1)
    result["disk_slo_working_set_pct"] = round(
        ctx.get("disk_slo_working_set", 0) * 100, 1
    )
    result["write_buffer_gib"] = round(ctx.get("write_buffer_gib", 0), 2)
    result["min_threshold_ctx"] = ctx.get("min_threshold", "N/A")
    result["req_disk_gib"] = round(req.disk_gib.mid, 1) if req.disk_gib else "N/A"
    result["req_mem_gib"] = round(req.mem_gib.mid, 1) if req.mem_gib else "N/A"

    # EBS drive details
    if cluster.attached_drives:
        drive = cluster.attached_drives[0]
        result["ebs_size_gib"] = drive.size_gib
        result["ebs_read_iops"] = drive.read_io_per_s
        result["ebs_write_iops"] = drive.write_io_per_s
        result["ebs_total_iops"] = (drive.read_io_per_s or 0) + (
            drive.write_io_per_s or 0
        )
        result["ebs_annual_cost_per_vol"] = round(drive.annual_cost, 2)
    elif cluster.instance.drive:
        drive = cluster.instance.drive
        result["local_disk_gib"] = drive.size_gib
        result["local_disk_read_iops"] = drive.read_io_per_s
        result["local_disk_write_iops"] = drive.write_io_per_s

    return result


# ──────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────

SCENARIOS: list[tuple[str, CapacityDesires, dict[str, Any]]] = []

# 1. Small dataset, high QPS — Why volumes are tiny
SCENARIOS.append(
    (
        "small_high_qps",
        CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(100_000),
                estimated_write_per_second=certain_int(100_000),
                estimated_mean_read_size_bytes=certain_int(256),
                estimated_mean_write_size_bytes=certain_int(256),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10),
            ),
        ),
        {},
    )
)

# 2. Medium dataset, moderate QPS — Typical workload
SCENARIOS.append(
    (
        "medium_moderate",
        CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(10_000),
                estimated_write_per_second=certain_int(10_000),
                estimated_mean_read_size_bytes=certain_int(1024),
                estimated_mean_write_size_bytes=certain_int(1024),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(500),
            ),
        ),
        {},
    )
)

# 3. Large dataset, read-heavy — Memory tax from EBS latency
SCENARIOS.append(
    (
        "large_read_heavy",
        CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(100_000),
                estimated_write_per_second=certain_int(1_000),
                estimated_mean_read_size_bytes=certain_int(1024),
                estimated_mean_write_size_bytes=certain_int(1024),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(2000),
            ),
        ),
        {},
    )
)

# 4. Large dataset, write-heavy — Write amplification stress
SCENARIOS.append(
    (
        "large_write_heavy",
        CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1_000),
                estimated_write_per_second=certain_int(500_000),
                estimated_mean_read_size_bytes=certain_int(1024),
                estimated_mean_write_size_bytes=certain_int(4096),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(300),
            ),
        ),
        {"copies_per_region": 2},
    )
)

# 5. Very large dataset, low QPS — Pure storage scenario
SCENARIOS.append(
    (
        "very_large_low_qps",
        CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1_000),
                estimated_write_per_second=certain_int(1_000),
                estimated_mean_read_size_bytes=certain_int(1024),
                estimated_mean_write_size_bytes=certain_int(1024),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10_000),
            ),
        ),
        {},
    )
)

# 6. Extreme writes — IOPS saturation
SCENARIOS.append(
    (
        "extreme_writes",
        CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1_000),
                estimated_write_per_second=certain_int(1_000_000),
                estimated_mean_read_size_bytes=certain_int(1024),
                estimated_mean_write_size_bytes=certain_int(4096),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(100),
            ),
        ),
        {"max_regional_size": 96 * 2},
    )
)


def print_separator(char: str = "═", width: int = 120) -> None:
    print(char * width)


def print_scenario_header(name: str, desires: CapacityDesires) -> None:
    qp = desires.query_pattern
    ds = desires.data_shape
    print(f"\n{'─' * 120}")
    print(f"SCENARIO: {name}")
    print(
        f"  Reads/s: {int(qp.estimated_read_per_second.mid):,}  "
        f"Writes/s: {int(qp.estimated_write_per_second.mid):,}  "
        f"Data: {int(ds.estimated_state_size_gib.mid):,} GiB  "
        f"Read size: {int(qp.estimated_mean_read_size_bytes.mid)} B  "
        f"Write size: {int(qp.estimated_mean_write_size_bytes.mid)} B"
    )
    print(f"{'─' * 120}")


def print_result(r: dict[str, Any], label: str) -> None:
    if "error" in r:
        print(f"  [{label}] ERROR: {r['error']}")
        return

    storage_info = ""
    if "ebs_size_gib" in r:
        storage_info = (
            f"EBS: {r['ebs_size_gib']} GiB  "
            f"R:{r['ebs_read_iops']} W:{r['ebs_write_iops']} IOPS  "
            f"(${r['ebs_annual_cost_per_vol']:,.0f}/vol/yr)"
        )
    elif "local_disk_gib" in r:
        storage_info = f"Local: {r['local_disk_gib']:,} GiB"

    print(
        f"  [{label:5s}] {r['instance']:16s} x {r['count_per_zone']:3d}/zone "
        f"({r['cpu']:2d}cpu, {r['ram_gib']:5.0f}GB RAM)  "
        f"RF={r['rf']}  "
        f"${r['total_annual_cost']:>10,.0f}/yr  "
        f"{storage_info}"
    )
    print(
        f"          Heap: {r['heap_gib']} GiB  "
        f"Write%: {r['write_buffer_pct']}  "
        f"Table%: {r['table_buffer_pct']}  "
        f"Compaction min_thresh: {r['compaction_min_threshold']}  "
        f"WorkingSet: {r['working_set_pct']}%"
    )
    print(
        f"          Req disk: {r['req_disk_gib']} GiB/zone  "
        f"Req mem: {r['req_mem_gib']} GiB/zone  "
        f"Write buffer: {r['write_buffer_gib']} GiB/zone  "
        f"RPS WS: {r['rps_working_set_pct']}%  "
        f"Disk SLO WS: {r['disk_slo_working_set_pct']}%"
    )


ResultRow = tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]


def print_cost_comparison(
    ebs_result: dict[str, Any],
    local_result: dict[str, Any],
    auto_result: dict[str, Any],
) -> None:
    """Print cost comparison between EBS and local results."""
    ebs_cost = ebs_result.get("total_annual_cost", 0)
    local_cost = local_result.get("total_annual_cost", 0)
    if ebs_cost and local_cost:
        ratio = ebs_cost / local_cost
        cheaper = "EBS" if ratio < 1 else "LOCAL"
        savings_pct = abs(1 - ratio) * 100
        print(
            f"          >>> EBS/LOCAL cost ratio: "
            f"{ratio:.2f}x -- "
            f"{cheaper} is {savings_pct:.0f}% cheaper"
        )
        auto_inst = auto_result.get("instance", "?")
        auto_has_ebs = "ebs_size_gib" in auto_result
        storage = "EBS" if auto_has_ebs else "local disk"
        print(f"          >>> AUTO picks: {auto_inst} ({storage})")


def print_summary_table(
    all_results: list[ResultRow],
) -> None:
    """Print the summary comparison table."""
    print(f"\n\n{'=' * 130}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'=' * 130}")
    header = (
        f"{'Scenario':<22s} | "
        f"{'EBS Instance':>14s} x{'#':>3s} | "
        f"{'Vol GiB':>7s} {'R IOPS':>7s} "
        f"{'W IOPS':>7s} | "
        f"{'EBS $/yr':>11s} | "
        f"{'Local Instance':>14s} x{'#':>3s} | "
        f"{'Local $/yr':>11s} | "
        f"{'Ratio':>5s} | "
        f"{'AUTO picks':>14s}"
    )
    print(header)
    print("-" * 130)

    for name, ebs, local, auto in all_results:
        ebs_inst = ebs.get("instance", "ERR")
        ebs_cnt = ebs.get("count_per_zone", 0)
        vol_gib = ebs.get("ebs_size_gib", "-")
        r_iops = ebs.get("ebs_read_iops", "-")
        w_iops = ebs.get("ebs_write_iops", "-")
        ebs_cost = ebs.get("total_annual_cost", 0)

        local_inst = local.get("instance", "ERR")
        local_cnt = local.get("count_per_zone", 0)
        local_cost = local.get("total_annual_cost", 0)

        ratio = f"{ebs_cost / local_cost:.2f}" if local_cost else "N/A"
        auto_pick = auto.get("instance", "ERR")

        print(
            f"{name:<22s} | "
            f"{ebs_inst:>14s} x{ebs_cnt:>3d} | "
            f"{str(vol_gib):>7s} "
            f"{str(r_iops):>7s} "
            f"{str(w_iops):>7s} | "
            f"${ebs_cost:>10,.0f} | "
            f"{local_inst:>14s} x{local_cnt:>3d} | "
            f"${local_cost:>10,.0f} | "
            f"{ratio:>5s} | "
            f"{auto_pick:>14s}"
        )

    print(f"{'=' * 130}")


def print_key_analysis(
    all_results: list[ResultRow],
) -> None:
    """Print key analysis findings for each scenario."""
    print("\n\nKEY ANALYSIS POINTS")
    print("=" * 80)
    for name, ebs, local, auto in all_results:
        if "error" in ebs or "error" in local:
            print(f"\n--- {name} ---")
            if "error" in ebs:
                print(f"  EBS ERROR: {ebs['error']}")
            if "error" in local:
                print(f"  LOCAL ERROR: {local['error']}")
            continue
        _print_scenario_analysis(name, ebs, local, auto)


def _print_scenario_analysis(
    name: str,
    ebs: dict[str, Any],
    local: dict[str, Any],
    auto: dict[str, Any],
) -> None:
    """Print analysis for a single scenario."""
    print(f"\n--- {name} ---")

    # Why volumes are small
    vol_size = ebs.get("ebs_size_gib", None)
    if vol_size is not None and vol_size <= 200:
        cnt = ebs["count_per_zone"]
        print(
            f"  SMALL VOLUME: {vol_size} GiB "
            f"(req disk: {ebs['req_disk_gib']} GiB/zone, "
            f"count: {cnt}/zone)"
        )
        if ebs["req_disk_gib"] != "N/A":
            per_node = ebs["req_disk_gib"] / cnt
            print(
                f"  Per-node data: {per_node:.0f} GiB "
                f"-> next_n(max(1, io, space), 100) "
                f"= {vol_size} GiB"
            )

    # Write amplification
    if ebs.get("compaction_min_threshold", 4) > 4:
        thresh = ebs["compaction_min_threshold"]
        print(
            f"  HIGH COMPACTION THRESHOLD: {thresh} "
            f"(normal=4, indicates write pressure)"
        )

    # Memory tax comparison
    ebs_ws = ebs.get("working_set_pct", 0)
    local_ws = local.get("working_set_pct", 0)
    if ebs_ws != local_ws:
        delta = ebs_ws - local_ws
        print(
            f"  WORKING SET: EBS={ebs_ws}% "
            f"vs LOCAL={local_ws}% "
            f"(EBS needs {delta:.1f}% more in RAM)"
        )

    # Cost comparison
    ebs_cost = ebs.get("total_annual_cost", 0)
    local_cost = local.get("total_annual_cost", 0)
    if ebs_cost and local_cost:
        diff = ebs_cost - local_cost
        ratio = ebs_cost / local_cost
        print(
            f"  COST: EBS ${ebs_cost:,.0f} "
            f"vs Local ${local_cost:,.0f} "
            f"(diff: ${diff:+,.0f}, "
            f"ratio: {ratio:.2f}x)"
        )

    # IOPS saturation
    total_iops = ebs.get("ebs_total_iops", 0)
    if total_iops > 10000:
        pct = total_iops / 16000 * 100
        print(
            f"  IOPS PRESSURE: {total_iops:,} "
            f"/ 16,000 max per vol "
            f"({pct:.0f}% of gp3 max)"
        )

    # Auto mode insight
    auto_inst = auto.get("instance", "?")
    auto_has_ebs = "ebs_size_gib" in auto
    storage = "with EBS" if auto_has_ebs else "with local disk"
    auto_cost = auto.get("total_annual_cost", 0)
    print(f"  AUTO MODE: picks {auto_inst} ({storage}) -- ${auto_cost:,.0f}/yr")


def main() -> None:
    print_separator()
    print("EBS DISK INVESTIGATION")
    print_separator()

    all_results: list[ResultRow] = []

    for name, desires, extra_args in SCENARIOS:
        print_scenario_header(name, desires)

        ebs_result = run_scenario(
            name,
            desires,
            mode="ebs",
            extra_args={**extra_args},
        )
        local_result = run_scenario(
            name,
            desires,
            mode="local",
            extra_args={**extra_args},
        )
        auto_result = run_scenario(
            name,
            desires,
            mode="auto",
            extra_args={**extra_args},
        )

        print_result(ebs_result, "EBS")
        print_result(local_result, "LOCAL")
        print_result(auto_result, "AUTO")

        print_cost_comparison(
            ebs_result,
            local_result,
            auto_result,
        )

        all_results.append((name, ebs_result, local_result, auto_result))

    print_summary_table(all_results)
    print_key_analysis(all_results)


if __name__ == "__main__":
    main()
