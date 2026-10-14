"""Prompt for setup, run the agent's BFS/DFS search, animate the search."""
import numpy as np

from agent import bfs_search, dfs_search
from maze import Maze
from maze_view import MazeView

WALL_PROB = 0.25
# If True, DFS animates the agent's physical footsteps including
# backtracking; if False (default), both BFS and DFS walk the
# reconstructed route (optimal within the searched nodes) after the
# exploration phase shows the search itself.
BACKTRACK_WALK = False


def ask_int(prompt, minimum=1):
    while True:
        try:
            value = int(input(prompt))
            if value < minimum:
                print(f"  Please enter an integer >= {minimum}.")
                continue
            return value
        except ValueError:
            print("  Please enter a valid integer.")


def ask_pos(name, nrows, ncols, rng):
    while True:
        raw = input(f"{name} position as 'row col' "
                    f"(0-{nrows - 1}, 0-{ncols - 1}) [blank = random]: ").strip()
        if not raw:
            pos = (int(rng.integers(nrows)), int(rng.integers(ncols)))
            print(f"  Random {name.lower()}: {pos[0]} {pos[1]}")
            return pos
        try:
            r, c = map(int, raw.split())
        except ValueError:
            print("  Enter two integers separated by a space, e.g. '0 0'.")
            continue
        if not (0 <= r < nrows and 0 <= c < ncols):
            print(f"  Out of range for a {nrows}x{ncols} maze.")
            continue
        return (r, c)


def ask_wall_prob(default=WALL_PROB):
    while True:
        raw = input(f"Wall probability 0.1-0.9 [default {default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number, e.g. 0.25.")
            continue
        if not 0.1 <= value <= 0.9:
            print("  Must be between 0.1 and 0.9.")
            continue
        return value


def ask_algorithm():
    while True:
        raw = input("Search algorithm (bfs/dfs) [default bfs]: ").strip().lower()
        if raw in ("", "bfs"):
            return "bfs"
        if raw == "dfs":
            return "dfs"
        print("  Please enter 'bfs' or 'dfs'.")


def main():
    print("=== Maze Setup ===")
    nrows = ask_int("Height (rows): ")
    ncols = ask_int("Width (cols): ")
    rng = np.random.default_rng()
    start = ask_pos("Start", nrows, ncols, rng)
    goal = ask_pos("Goal", nrows, ncols, rng)
    while goal == start:
        goal = (int(rng.integers(nrows)), int(rng.integers(ncols)))
        print(f"  Goal matched start, re-rolled to: {goal[0]} {goal[1]}")
    wall_prob = ask_wall_prob()
    algo = ask_algorithm()

    print(f"\nGenerating a solvable {nrows}x{ncols} maze "
          f"(start={start}, goal={goal}, walls~{wall_prob})...")
    maze, _ = Maze.generate_solvable(nrows, ncols, wall_prob,
                                     start=start, goal=goal)
    print("Maze ready. Running agent search...")

    search = bfs_search if algo == "bfs" else dfs_search
    result = search(maze)
    if BACKTRACK_WALK and getattr(result, "walk", None):
        walk = result.walk
        walk_desc = "physical walk (backtracking incl.)"
    else:
        walk = result.path
        walk_desc = "reconstructed route"
    print(f"{algo.upper()}: explored {len(result.explored)} cells, "
          f"{walk_desc} length {len(walk) - 1} steps.")

    view = MazeView(maze)
    view.show_search(result.explored, walk, parents=result.parents, loop=True)
    view.show()


if __name__ == "__main__":
    main()
