"""Agent confined to 4-directional movement with 4-directional sensing.

The agent never sees the full grid: it may only
  - move up/down/left/right into free cells (never into walls), and
  - sense the 4 orthogonal neighbours of its current cell.

Search algorithms (BFS / DFS) drive the agent using only those two
capabilities, and record the exploration order so the visualization can
show the search the agent actually performed (not just the optimal path).
"""
from collections import deque
import heapq
import math

# The only 4 motions allowed: up, down, left, right.
DIRS = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# All 8 motions: the 4 orthogonal plus the 4 diagonals.
DIRS8 = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "up_right": (1, 1),
    "up_left": (1, -1),
    "down_right": (-1, 1),
    "down_left": (-1, -1),
}


class SearchResult:
    """Outcome of a search: what was explored, and the final route found."""
    def __init__(self, explored, path, parents=None):
        self.explored = explored  # cells in the order first expanded/visited
        self.path = path          # route start -> goal (None if not found)
        self.parents = parents or {}  # child -> parent discovery tree


class Agent:
    """Agent that senses and moves in the maze using 4-dir sensing."""

    def __init__(self, maze):
        """Initialize agent at maze start."""
        self.maze = maze
        self.pos = maze.start


def bfs_search(maze):
    """Breadth-first search using only the agent's 4-dir sensing.

    Returns SearchResult(explored in expansion order, shortest path or None).
    """
    agent = Agent(maze)
    agent.pos = maze.start
    prev = {maze.start: None}
    explored = []
    queue = deque([maze.start])
    while queue:
        cur = queue.popleft()
        agent.pos = cur  # agent "visits" via sensing, never teleports blindly
        explored.append(cur)
        if cur == maze.goal:
            break
        for nb in _sensed_free(maze, cur):
            if nb not in prev:
                prev[nb] = cur
                queue.append(nb)
    path = _reconstruct(prev, maze.start, maze.goal)
    return SearchResult(explored, path, parents=prev)


def dfs_search(maze):
    """Depth-first search using only the agent's 4-dir sensing.

    Returns SearchResult(explored in visit order, path or None).
    Explored order here IS the agent's walk order (backtracking included
    is reported via `walk` below).
    """
    agent = Agent(maze)
    prev = {maze.start: None}
    explored = []
    walk = [maze.start]  # physical walk, backtracking steps included
    stack = [maze.start]
    visited = {maze.start}
    while stack:
        cur = stack.pop()
        agent.pos = cur
        explored.append(cur)
        if cur == maze.goal:
            break
        for nb in _sensed_free(maze, cur):
            if nb not in visited:
                visited.add(nb)
                prev[nb] = cur
                stack.append(nb)
    # Rebuild the physical walk with backtracking: walk from each explored
    # node back along `prev` when DFS retreats.
    walk = _dfs_walk(prev, explored, maze.start)
    path = _reconstruct(prev, maze.start, maze.goal)
    result = SearchResult(explored, path, parents=prev)
    result.walk = walk
    return result


def _sensed_free(maze, pos):
    """Free orthogonal neighbours of pos (what an agent there could see)."""
    r, c = pos
    out = []
    for dr, dc in DIRS.values():
        nb = (r + dr, c + dc)
        if maze.is_free(nb):
            out.append(nb)
    return out


def _reconstruct(prev, start, goal):
    """Rebuild path from goal back to start using parent map."""
    if goal not in prev:
        return None
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def _dfs_walk(prev, explored, start):
    """Physical walk incl. backtracking steps between explored nodes."""
    if not explored:
        return []
    walk = [explored[0]]
    for nxt in explored[1:]:
        # climb from current walk-end up via prev until adjacent to nxt
        while not _adjacent(walk[-1], nxt):
            parent = prev.get(walk[-1])
            if parent is None:  # disconnected; should not happen
                break
            walk.append(parent)
        walk.append(nxt)
    return walk


def _adjacent(a, b):
    """Check if two cells are orthogonally adjacent."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# ---------------------------------------------------------------------------
# Heuristics (all admissible for unit-step-cost grid search)
# ---------------------------------------------------------------------------


def manhattan(a, b):
    """Manhattan (L1) distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """Euclidean (L2) distance between two cells."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a, b):
    """Chebyshev (L-infinity) distance between two cells.

    Admissible for 8-directional movement with unit step cost (a diagonal step
    costs 1, same as orthogonal), so it never overestimates the true cost.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
}


def _neighbors(maze, pos, diagonal):
    """Free neighbours of pos, honoring the movement mode."""
    return maze.neighbors(pos, diagonal=diagonal)


# ---------------------------------------------------------------------------
# Iterative Deepening DFS
# ---------------------------------------------------------------------------


def id_dfs_search(maze, heuristic="manhattan", diagonal=False):
    """Iterative Deepening DFS.

    Repeatedly runs depth-limited DFS with increasing depth limits until the
    goal is found or all reachable nodes are exhausted. Returns a SearchResult
    whose `explored` is the concatenation of each iteration's expansion order
    and whose `path` is the reconstructed route (or None if unreachable).
    """
    start = maze.start
    goal = maze.goal
    if not maze.is_free(start) or not maze.is_free(goal):
        return SearchResult([], None, parents={})

    h = HEURISTICS[heuristic]
    parents = {start: None}
    explored = []
    all_visited = {start}
    depth = 0

    while True:
        result = _depth_limited_dfs(maze, start, goal, depth, diagonal,
                                     parents, all_visited)
        explored.extend(result.explored)
        if result.path is not None:
            return SearchResult(explored, result.path, parents=parents)
        if result.exhausted:
            # Depth limit was sufficient to reach all nodes; goal unreachable.
            return SearchResult(explored, None, parents=parents)
        depth += 1


def _depth_limited_dfs(maze, start, goal, limit, diagonal, parents,
                        all_visited):
    """Single depth-limited DFS iteration.

    Returns an _IterResult. `exhausted` is True only if the search explored
    every reachable node without hitting the depth limit (goal unreachable).
    """
    explored = []
    visited = {start}
    stack = [(start, 0)]
    found = False
    cutoff = False  # True if we hit the depth limit on some branch

    while stack:
        node, d = stack.pop()
        explored.append(node)
        all_visited.add(node)
        if node == goal:
            found = True
            break
        if d < limit:
            for nb in _neighbors(maze, node, diagonal):
                if nb not in visited:
                    visited.add(nb)
                    parents[nb] = node
                    stack.append((nb, d + 1))
        else:
            cutoff = True  # reached depth limit, more nodes may exist below

    path = _reconstruct(parents, start, goal) if found else None
    # Exhausted only if we didn't find the goal AND didn't cut off any branch.
    exhausted = (not found) and (not cutoff)
    return _IterResult(explored=explored, path=path, exhausted=exhausted)


class _IterResult:
    def __init__(self, explored, path, exhausted):
        self.explored = explored
        self.path = path
        self.exhausted = exhausted


# ---------------------------------------------------------------------------
# Greedy Best-First Search
# ---------------------------------------------------------------------------


def best_first_search(maze, heuristic="manhattan", diagonal=False):
    """Greedy Best-First Search.

    Priority queue ordered by heuristic h(n) only.
    """
    return _priority_search(maze, heuristic, diagonal, greedy=True)


# ---------------------------------------------------------------------------
# A* Search
# ---------------------------------------------------------------------------


def astar_search(maze, heuristic="manhattan", diagonal=False):
    """A* Search.

    Priority queue ordered by f(n) = g(n) + h(n).
    """
    return _priority_search(maze, heuristic, diagonal, greedy=False)


def _priority_search(maze, heuristic, diagonal, greedy):
    """Shared core for Greedy Best-First and A*."""
    start = maze.start
    goal = maze.goal
    if not maze.is_free(start) or not maze.is_free(goal):
        return SearchResult([], None, parents={})

    h = HEURISTICS[heuristic]
    parents = {start: None}
    g = {start: 0}
    explored = []
    counter = 0
    start_priority = 0 if greedy else h(start, goal)
    pq = [(start_priority, counter, start)]
    in_pq = {start}

    while pq:
        _, _, node = heapq.heappop(pq)
        in_pq.discard(node)
        explored.append(node)
        if node == goal:
            return SearchResult(explored, _reconstruct(parents, start, goal),
                                parents=parents)
        for nb in _neighbors(maze, node, diagonal):
            tentative_g = g[node] + 1
            if nb not in g or tentative_g < g[nb]:
                g[nb] = tentative_g
                parents[nb] = node
                priority = tentative_g if greedy else tentative_g + h(nb, goal)
                if nb not in in_pq:
                    counter += 1
                    heapq.heappush(pq, (priority, counter, nb))
                    in_pq.add(nb)
                else:
                    counter += 1
                    heapq.heappush(pq, (priority, counter, nb))

    return SearchResult(explored, None, parents=parents)
