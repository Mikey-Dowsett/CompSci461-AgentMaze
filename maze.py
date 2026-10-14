"""Maze representation: n x m array, 0 = free, 1 = wall."""
from collections import deque

import numpy as np


class Maze:
    def __init__(self, grid, start=(0, 0), goal=None):
        self.grid = np.array(grid, dtype=int)
        self.nrows, self.ncols = self.grid.shape
        self.start = start
        self.goal = goal if goal is not None else (self.nrows - 1, self.ncols - 1)
        if self.grid[self.start] == 1:
            raise ValueError(f"Start {self.start} is inside a wall")
        if self.grid[self.goal] == 1:
            raise ValueError(f"Goal {self.goal} is inside a wall")

    def is_free(self, pos):
        r, c = pos
        return (0 <= r < self.nrows and 0 <= c < self.ncols
                and self.grid[r, c] == 0)

    def neighbors(self, pos):
        r, c = pos
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if self.is_free(nb):
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
        return self.find_path(start, goal) is not None

    @staticmethod
    def random(nrows, ncols, wall_prob=0.25, seed=None,
               start=(0, 0), goal=None):
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
