"""One-search subprocess used to isolate runtime and peak-memory readings."""
import json
import sys
import time
import tracemalloc

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None

from agent import (astar_search, best_first_search, bfs_search, dfs_search,
                   id_dfs_search)
from maze import Maze


SEARCHES = {
    "bfs": bfs_search,
    "dfs": dfs_search,
    "iddfs": id_dfs_search,
    "best_first": best_first_search,
    "astar": astar_search,
}


def _peak_rss_bytes():
    if resource is not None:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            return int(value)
        return int(value * 1024)
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except ImportError:
        return None


def run(request):
    maze = Maze(request["grid"], start=tuple(request["start"]),
                goal=tuple(request["goal"]))
    algorithm = request["algorithm"]
    search = SEARCHES[algorithm]
    kwargs = {"capture_trace": request.get("capture_trace", False)}
    if algorithm in {"iddfs", "best_first", "astar"}:
        kwargs["heuristic"] = request.get("heuristic", "manhattan")
    if algorithm in {"bfs", "dfs", "iddfs", "best_first", "astar"}:
        kwargs["diagonal"] = request.get("diagonal", False)

    tracemalloc.start()
    started = time.perf_counter_ns()
    result = search(maze, **kwargs)
    runtime_ns = time.perf_counter_ns() - started
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = {
        "algorithm": algorithm,
        "runtime_ns": runtime_ns,
        "python_peak_bytes": python_peak_bytes,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "metrics": result.metrics.as_dict(),
        "path": result.path,
        "solution_depth": result.solution_depth,
        "path_cost": result.path_cost,
        "explored": result.explored if request.get("capture_trace") else None,
        "parents": [[child, parent] for child, parent in result.parents.items()]
            if request.get("capture_trace") else None,
        "events": result.events if request.get("capture_trace") else None,
        "walk": getattr(result, "walk", None)
            if request.get("capture_trace") else None,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    run(json.load(sys.stdin))
