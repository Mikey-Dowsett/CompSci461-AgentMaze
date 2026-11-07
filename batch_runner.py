"""Reproducible search profiling campaigns and annotated report export."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from maze import Maze


DEFAULT_SIZES = (10, 15, 20, 30)
DEFAULT_DENSITIES = (0.08, 0.16, 0.24)
DEFAULT_REPETITIONS = 5


def _method_configurations():
    configs = []
    for algorithm in ("bfs", "dfs", "iddfs"):
        for movement in ("orthogonal", "diagonal"):
            configs.append({"algorithm": algorithm, "movement": movement,
                            "heuristic": None})
    for algorithm in ("best_first", "astar"):
        for movement in ("orthogonal", "diagonal"):
            for heuristic in ("manhattan", "euclidean", "chebyshev"):
                configs.append({"algorithm": algorithm, "movement": movement,
                                "heuristic": heuristic})
    return configs


def run_profiled(maze, configuration, capture_trace=False):
    """Run one search in a fresh process and return its measurements."""
    request = {
        "grid": maze.grid.tolist(),
        "start": list(maze.start),
        "goal": list(maze.goal),
        "algorithm": configuration["algorithm"],
        "movement": configuration["movement"],
        "diagonal": configuration["movement"] == "diagonal",
        "heuristic": configuration.get("heuristic") or "manhattan",
        "capture_trace": capture_trace,
    }
    worker = Path(__file__).with_name("profile_worker.py")
    completed = subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Search worker failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid search-worker output: {completed.stdout!r}") from error


def _aggregate(runs):
    groups = {}
    for row in runs:
        key = (row["graph_id"], row["algorithm"], row["movement"],
               row["heuristic"])
        groups.setdefault(key, []).append(row)

    summaries = []
    numeric_fields = (
        "path_length", "solution_depth", "runtime_ns", "python_peak_bytes",
        "process_peak_rss_bytes", "nodes_generated", "nodes_expanded",
        "branching_avg", "branching_max", "peak_frontier",
        "peak_frontier_plus_explored",
    )
    for (graph_id, algorithm, movement, heuristic), members in groups.items():
        first = members[0]
        summary = {
            "graph_id": graph_id,
            "rows": len(members),
            "algorithm": algorithm,
            "movement": movement,
            "heuristic": heuristic,
            "size": first["size"],
            "density": first["density"],
            "seed": first["seed"],
            "solved_runs": sum(row["path_length"] is not None for row in members),
        }
        for field in numeric_fields:
            values = [row[field] for row in members if row[field] is not None]
            summary[f"{field}_mean"] = sum(values) / len(values) if values else None
            summary[f"{field}_best"] = min(values) if values else None
            summary[f"{field}_worst"] = max(values) if values else None
        summaries.append(summary)
    return summaries


def run_campaign(base_seed=70804, repetitions=DEFAULT_REPETITIONS,
                 sizes=DEFAULT_SIZES, densities=DEFAULT_DENSITIES,
                 progress=None, cancel_event=None, output_dir=None):
    """Run the default matched-maze protocol and persist JSON/CSV reports."""
    if repetitions < 5:
        raise ValueError("Batch comparisons require at least 5 repetitions.")
    if not sizes or not densities:
        raise ValueError("At least one size and one density are required.")

    started = datetime.now(timezone.utc)
    configurations = _method_configurations()
    graph_specs = [(size, density) for size in sizes for density in densities]
    total = len(graph_specs) * len(configurations) * repetitions
    completed_count = 0
    raw_runs = []
    graph_metadata = []
    cancelled = False

    for graph_index, (size, density) in enumerate(graph_specs):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        seed = int(base_seed) + graph_index
        graph_id = f"{size}x{size}-d{int(round(density * 100)):02d}-s{seed}"
        maze, _ = Maze.generate_solvable(size, size, density, seed=seed)
        actual_density = float((maze.grid == 1).mean())
        graph_metadata.append({
            "graph_id": graph_id,
            "rows": size,
            "cols": size,
            "obstacle_density_target": density,
            "obstacle_density_actual": actual_density,
            "seed": seed,
            "start": list(maze.start),
            "goal": list(maze.goal),
            "grid": maze.grid.tolist(),
        })

        for config in configurations:
            for repetition in range(1, repetitions + 1):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                result = run_profiled(maze, config)
                metrics = result["metrics"]
                path = result["path"]
                raw_runs.append({
                    "graph_id": graph_id,
                    "size": size,
                    "density": density,
                    "actual_density": actual_density,
                    "seed": seed,
                    "algorithm": config["algorithm"],
                    "movement": config["movement"],
                    "heuristic": config["heuristic"],
                    "repetition": repetition,
                    "path_length": len(path) - 1 if path else None,
                    "solution_depth": result["solution_depth"],
                    "path_cost": result["path_cost"],
                    "runtime_ns": result["runtime_ns"],
                    "python_peak_bytes": result["python_peak_bytes"],
                    "process_peak_rss_bytes": result["process_peak_rss_bytes"],
                    **metrics,
                })
                completed_count += 1
                if progress is not None:
                    progress({
                        "completed": completed_count,
                        "total": total,
                        "graph_id": graph_id,
                        "algorithm": config["algorithm"],
                        "repetition": repetition,
                    })
            if cancelled:
                break
        if cancelled:
            break

    summaries = _aggregate(raw_runs)
    finished = datetime.now(timezone.utc)
    output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "batch_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = finished.strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"search_batch_{stamp}.json"
    csv_path = output_dir / f"search_batch_{stamp}_runs.csv"
    metadata = {
        "title": "Maze search profiling campaign",
        "protocol_version": 1,
        "status": "cancelled" if cancelled else "completed",
        "started_seed": int(base_seed),
        "started_at_utc": started.isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "sizes": list(sizes),
        "obstacle_densities": list(densities),
        "repetitions_per_configuration": repetitions,
        "graph_count": len(graph_metadata),
        "method_configuration_count": len(configurations),
        "planned_run_count": total,
        "completed_run_count": completed_count,
        "metric_definitions": {
            "runtime_ns": "perf_counter_ns around the search only; process startup and maze creation excluded",
            "python_peak_bytes": "tracemalloc peak allocated during the search call",
            "process_peak_rss_bytes": "isolated worker process high-water RSS, including interpreter/import and maze-setup baseline",
            "peak_frontier_plus_explored": "peak frontier entries plus unique states expanded in the search (ID-DFS uses the current iteration's expanded set)",
            "nodes_generated": "unique discovered states, except ID-DFS counts repeated generation across depth limits",
            "branching_avg": "mean legal neighbors per expanded node, including already discovered neighbors",
            "path_length": "number of moves in the returned path; diagonal moves cost one",
            "best_worst": "minimum and maximum among repetitions; summaries also include the mean",
        },
        "finished_at_utc": finished.isoformat(),
        "graphs": graph_metadata,
        "runs": raw_runs,
        "summaries": summaries,
        "csv_path": str(csv_path),
    }
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if raw_runs:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            if raw_runs:
                writer = csv.DictWriter(handle, fieldnames=list(raw_runs[0].keys()))
                writer.writeheader()
                writer.writerows(raw_runs)
    return {"report_path": str(report_path), "csv_path": str(csv_path),
            "summaries": summaries, "runs": raw_runs,
            "completed": completed_count, "total": total,
            "cancelled": cancelled}


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible maze-search batch.")
    parser.add_argument("--config", help="JSON file overriding seed, sizes, densities, repetitions, or output_dir")
    args = parser.parse_args()
    options = {}
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            options = json.load(handle)
        allowed = {"base_seed", "sizes", "densities", "repetitions", "output_dir"}
        unknown = set(options) - allowed
        if unknown:
            parser.error(f"Unsupported config keys: {', '.join(sorted(unknown))}")
    def report_progress(update):
        done = update["completed"]
        if done == 1 or done % 20 == 0 or done == update["total"]:
            print(f"Progress: {done}/{update['total']} runs", flush=True)

    result = run_campaign(progress=report_progress, **options)
    print(f"{result['completed']}/{result['total']} runs saved")
    print(f"JSON report: {result['report_path']}")
    print(f"Raw CSV: {result['csv_path']}")


if __name__ == "__main__":
    main()
