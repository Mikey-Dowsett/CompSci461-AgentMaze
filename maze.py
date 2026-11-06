"""Maze representation: n x m array, 0 = free, 1 = wall."""
from collections import deque

import numpy as np


class Maze:
    def __init__(self, grid, start=(0, 0), goal=None):
        """Create a maze from a grid array (0=free, 1=wall)."""
        self.grid = np.array(grid, dtype=int)
        self.nrows, self.ncols = self.grid.shape
        self.start = start
        self.goal = goal if goal is not None else (self.nrows - 1, self.ncols - 1)
        if self.grid[self.start] == 1:
            raise ValueError(f"Start {self.start} is inside a wall")
        if self.grid[self.goal] == 1:
            raise ValueError(f"Goal {self.goal} is inside a wall")

    def is_free(self, pos):
        """Return True if pos is in-bounds and not a wall."""
        r, c = pos
        return (0 <= r < self.nrows and 0 <= c < self.ncols
                and self.grid[r, c] == 0)

    def neighbors(self, pos, diagonal=False):
        """Yield free neighbours of pos.

        If diagonal is False (default), only the 4 orthogonal neighbours are
        yielded. If True, the 8 surrounding cells are yielded; diagonal moves
        are only allowed when the target cell is free and the two corner-adjacent
        orthogonal cells are also free (so the agent cannot cut the corner
        through a wall).
        """
        r, c = pos
        orth = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for dr, dc in orth:
            nb = (r + dr, c + dc)
            if self.is_free(nb):
                yield nb
        if diagonal:
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                nb = (r + dr, c + dc)
                if (self.is_free(nb)
                        and self.is_free((r + dr, c))
                        and self.is_free((r, c + dc))):
                    yield nb

    def find_path(self, start=None, goal=None):
        """BFS shortest path from start to goal, or None if unreachable."""
        start = start if start is not None else self.start
        goal = goal if goal is not None else self.goal
        if not self.is_free(start) or not self.is_free(goal):
            return None
        prev = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for nb in self.neighbors(cur):
                if nb not in prev:
                    prev[nb] = cur
                    queue.append(nb)
        if goal not in prev:
            return None
        path, cur = [], goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    def has_path(self, start=None, goal=None):
        """Return True if a path exists from start to goal."""
        return self.find_path(start, goal) is not None

    @staticmethod
    def random(nrows, ncols, wall_prob=0.25, seed=None,
               start=(0, 0), goal=None):
        """Generate a random maze with given wall probability."""
        goal = goal if goal is not None else (nrows - 1, ncols - 1)
        rng = np.random.default_rng(seed)
        grid = (rng.random((nrows, ncols)) < wall_prob).astype(int)
        grid[start] = 0
        grid[goal] = 0
        return Maze(grid, start=start, goal=goal)

    @staticmethod
    def generate_solvable(nrows, ncols, wall_prob=0.25,
                          start=(0, 0), goal=None, max_tries=1000,
                          seed=None):
        """Keep generating random mazes until start can reach goal.

        Returns (maze, path); raises ValueError after max_tries failures.
        """
        goal = goal if goal is not None else (nrows - 1, ncols - 1)
        rng = np.random.default_rng(seed)
        for _ in range(max_tries):
            maze = Maze.random(nrows, ncols, wall_prob,
                               seed=rng.integers(2**63),
                               start=start, goal=goal)
            path = maze.find_path()
            if path is not None:
                return maze, path
        raise ValueError(
            f"No solvable maze found in {max_tries} tries "
            f"(size={nrows}x{ncols}, wall_prob={wall_prob}). "
            "Try a lower wall probability.")
