"""Agent confined to 4-directional movement with 4-directional sensing.

The agent never sees the full grid: it may only
  - move up/down/left/right into free cells (never into walls), and
  - sense the 4 orthogonal neighbours of its current cell.

Search algorithms (BFS / DFS) drive the agent using only those two
capabilities, and record the exploration order so the visualization can
show the search the agent actually performed (not just the optimal path).
"""
from collections import deque

# The only 4 motions allowed: up, down, left, right.
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class SearchResult:
    """Outcome of a search: what was explored, and the final route found."""
    def __init__(self, explored, path, parents=None):
        self.explored = explored  # cells in the order first expanded/visited
        self.path = path          # route start -> goal (None if not found)
        self.parents = parents or {}  # child -> parent discovery tree


class Agent:
    def __init__(self, maze):
        self.maze = maze
        self.pos = maze.start

    def sense(self):
        """Look up/down/left/right. Returns {name: 'wall'|'free'|'edge'}.

        'edge' = outside the maze bounds (treated as blocked).
        """
        out = {}
        for name, (dr, dc) in DIRS.items():
            nb = (self.pos[0] + dr, self.pos[1] + dc)
            r, c = nb
            if not (0 <= r < self.maze.nrows and 0 <= c < self.maze.ncols):
                out[name] = "edge"
            elif self.maze.grid[nb] == 1:
                out[name] = "wall"
            else:
                out[name] = "free"
        return out

    def free_neighbours(self):
        """Neighbour cells the agent can see are free (4-directional)."""
        return [nb for nb in self._visible_neighbours()
                if self.maze.is_free(nb)]

    def _visible_neighbours(self):
        r, c = self.pos
        return [(r + dr, c + dc) for dr, dc in DIRS.values()]

    def move(self, direction):
        """Step one cell in direction ('up'/'down'/'left'/'right').

        Raises ValueError if the target is a wall or out of bounds.
        """
        if direction not in DIRS:
            raise ValueError(f"Unknown direction {direction!r}; "
                             f"use one of {sorted(DIRS)}")
        dr, dc = DIRS[direction]
        target = (self.pos[0] + dr, self.pos[1] + dc)
        if not self.maze.is_free(target):
            raise ValueError(f"Agent cannot move {direction} into {target}: "
                             "wall or out of bounds")
        self.pos = target
        return self.pos


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
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
