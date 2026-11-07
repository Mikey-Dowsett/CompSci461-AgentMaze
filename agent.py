"""Search algorithms and shared results/metrics for the maze project."""
import heapq
import math
from collections import deque
from typing import Any


DIRS = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class SearchMetrics:
    """Counters shared by searches; frontier sizes count queued entries."""

    def __init__(self):
        self.nodes_generated = 0
        self.nodes_expanded = 0
        self.branching_total = 0
        self.branching_max = 0
        self.peak_frontier = 0
        self.peak_frontier_plus_explored = 0
        self._expanded_unique = set()

    @property
    def branching_avg(self):
        if not self.nodes_expanded:
            return 0.0
        return self.branching_total / self.nodes_expanded

    def record_frontier(self, frontier_size, explored):
        self.peak_frontier = max(self.peak_frontier, frontier_size)
        self.peak_frontier_plus_explored = max(
            self.peak_frontier_plus_explored,
            frontier_size + len(explored),
        )

    def record_expansion(self, node, legal_successors):
        self.nodes_expanded += 1
        self._expanded_unique.add(node)
        degree = len(legal_successors)
        self.branching_total += degree
        self.branching_max = max(self.branching_max, degree)

    def as_dict(self):
        return {
            "nodes_generated": self.nodes_generated,
            "nodes_expanded": self.nodes_expanded,
            "branching_avg": self.branching_avg,
            "branching_max": self.branching_max,
            "peak_frontier": self.peak_frontier,
            "peak_frontier_plus_explored": self.peak_frontier_plus_explored,
        }


class SearchResult:
    """Outcome of a search, including its trace and algorithmic counters."""

    def __init__(self, explored, path, parents=None, metrics=None, events=None):
        self.explored = explored
        self.path = path
        self.parents = parents or {}
        self.metrics = metrics or SearchMetrics()
        self.events = events or []
        self.solution_depth = len(path) - 1 if path else None
        self.path_cost = self.solution_depth
        self.walk = []


def _make_result(explored, path, parents, metrics, events):
    result = SearchResult(explored, path, parents, metrics, events)
    result.solution_depth = len(path) - 1 if path else None
    result.path_cost = result.solution_depth
    return result


def bfs_search(maze, diagonal=False, capture_trace=False):
    """Breadth-first search; diagonal steps have unit cost when enabled."""
    start, goal = maze.start, maze.goal
    parents = {start: None}
    explored = []
    expanded = set()
    queue = deque([start])
    metrics = SearchMetrics()
    metrics.nodes_generated = 1
    events = []
    metrics.record_frontier(len(queue), expanded)

    while queue:
        cur = queue.popleft()
        neighbors = list(maze.neighbors(cur, diagonal=diagonal))
        explored.append(cur)
        expanded.add(cur)
        metrics.record_expansion(cur, neighbors)
        if cur != goal:
            for nb in neighbors:
                if nb not in parents:
                    parents[nb] = cur
                    queue.append(nb)
                    metrics.nodes_generated += 1
        metrics.record_frontier(len(queue), expanded)
        if capture_trace:
            events.append({"type": "expand", "node": cur,
                           "frontier": len(queue), "frontier_nodes": list(queue),
                           "neighbors": neighbors})
        if cur == goal:
            break

    return _make_result(explored, _reconstruct(parents, start, goal),
                        parents, metrics, events)


def dfs_search(maze, diagonal=False, capture_trace=False):
    """Depth-first search; ``walk`` includes physical backtracking steps."""
    start, goal = maze.start, maze.goal
    parents = {start: None}
    explored = []
    expanded = set()
    stack = [start]
    visited = {start}
    metrics = SearchMetrics()
    metrics.nodes_generated = 1
    events = []
    metrics.record_frontier(len(stack), expanded)

    while stack:
        cur = stack.pop()
        neighbors = list(maze.neighbors(cur, diagonal=diagonal))
        explored.append(cur)
        expanded.add(cur)
        metrics.record_expansion(cur, neighbors)
        if cur != goal:
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    parents[nb] = cur
                    stack.append(nb)
                    metrics.nodes_generated += 1
        metrics.record_frontier(len(stack), expanded)
        if capture_trace:
            events.append({"type": "expand", "node": cur,
                           "frontier": len(stack), "frontier_nodes": list(stack),
                           "neighbors": neighbors})
        if cur == goal:
            break

    path = _reconstruct(parents, start, goal)
    result = _make_result(explored, path, parents, metrics, events)
    result.walk = _dfs_walk(parents, explored, maze, diagonal)
    return result


def id_dfs_search(maze, heuristic="manhattan", diagonal=False,
                  capture_trace=False):
    """Iterative deepening DFS; ID-DFS does not use a heuristic.

    Each depth-limited iteration tracks the best depth at which a state has
    been reached, allowing a later shallower route to reopen it. Repeated
    expansions and generation across limits are included in the metrics.
    """
    del heuristic
    start, goal = maze.start, maze.goal
    parents = {start: None}
    explored = []
    metrics = SearchMetrics()
    events = []
    limit = 0

    while True:
        best_depth = {start: 0}
        iteration_expanded = set()
        stack: list[tuple[Any, int, tuple[Any, ...]]] = [
            (start, 0, (start,))]
        metrics.nodes_generated += 1
        cutoff = False
        found_path = None
        while stack:
            node, depth, route = stack.pop()
            parents[node] = route[-2] if len(route) > 1 else None
            neighbors = list(maze.neighbors(node, diagonal=diagonal))
            explored.append(node)
            iteration_expanded.add(node)
            metrics.record_expansion(node, neighbors)
            if node == goal:
                found_path = route
            elif depth < limit:
                for nb in reversed(neighbors):
                    next_depth = depth + 1
                    if (nb not in route
                            and next_depth < best_depth.get(nb, math.inf)):
                        best_depth[nb] = next_depth
                        parents[nb] = node
                        stack.append((nb, next_depth, route + (nb,)))
                        metrics.nodes_generated += 1
            elif any(nb not in route and best_depth.get(nb, math.inf) > depth + 1
                     for nb in neighbors):
                cutoff = True
            metrics.record_frontier(len(stack), iteration_expanded)
            if capture_trace:
                events.append({"type": "expand", "node": node,
                               "frontier": len(stack),
                               "frontier_nodes": [item[0] for item in stack],
                               "depth_limit": limit, "neighbors": neighbors})
            if found_path is not None:
                break
        if found_path is not None:
            for parent, child in zip(found_path, found_path[1:]):
                parents[child] = parent
            return _make_result(explored, list(found_path), parents,
                                metrics, events)
        if not cutoff:
            return _make_result(explored, None, parents, metrics, events)
        limit += 1


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
}


def best_first_search(maze, heuristic="manhattan", diagonal=False,
                      capture_trace=False):
    """Greedy Best-First Search ordered by h(n)."""
    return _priority_search(maze, heuristic, diagonal, greedy=True,
                            capture_trace=capture_trace)


def astar_search(maze, heuristic="manhattan", diagonal=False,
                 capture_trace=False):
    """A* search ordered by g(n) + h(n)."""
    return _priority_search(maze, heuristic, diagonal, greedy=False,
                            capture_trace=capture_trace)


def _priority_search(maze, heuristic, diagonal, greedy, capture_trace=False):
    start, goal = maze.start, maze.goal
    h = HEURISTICS[heuristic]
    parents = {start: None}
    g_score = {start: 0}
    explored = []
    expanded = set()
    metrics = SearchMetrics()
    metrics.nodes_generated = 1
    events = []
    counter = 0
    frontier = [(h(start, goal), counter, start, 0)]
    metrics.record_frontier(1, expanded)

    while frontier:
        priority, _, node, entry_g = heapq.heappop(frontier)
        if entry_g != g_score.get(node):
            continue
        expanded.add(node)
        explored.append(node)
        neighbors = list(maze.neighbors(node, diagonal=diagonal))
        metrics.record_expansion(node, neighbors)
        if node != goal:
            for nb in neighbors:
                tentative_g = g_score[node] + 1
                if tentative_g < g_score.get(nb, math.inf):
                    first_discovery = nb not in g_score
                    g_score[nb] = tentative_g
                    parents[nb] = node
                    score = h(nb, goal) if greedy else tentative_g + h(nb, goal)
                    counter += 1
                    heapq.heappush(frontier, (score, counter, nb, tentative_g))
                    if first_discovery:
                        metrics.nodes_generated += 1
        metrics.record_frontier(len(frontier), expanded)
        if capture_trace:
            events.append({"type": "expand", "node": node,
                           "frontier": len(frontier),
                           "frontier_nodes": [entry[2] for entry in frontier],
                           "priority": priority, "neighbors": neighbors})
        if node == goal:
            return _make_result(explored, _reconstruct(parents, start, goal),
                                parents, metrics, events)

    return _make_result(explored, None, parents, metrics, events)


def _reconstruct(parents, start, goal):
    if goal not in parents:
        return None
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return path[::-1]


def _dfs_walk(parents, explored, maze, diagonal):
    """Return the legal DFS tree walk with backtracking between expansions."""
    if not explored:
        return []
    walk = [explored[0]]
    for nxt in explored[1:]:
        while nxt not in maze.neighbors(walk[-1], diagonal=diagonal):
            parent = parents.get(walk[-1])
            if parent is None:
                break
            walk.append(parent)
        walk.append(nxt)
    return walk
