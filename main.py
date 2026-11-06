"""Entry point: launch the Maze Explorer GUI.

Run this to start the graphical application:

    python main.py

The CLI has been retired; all search methods (BFS, DFS, ID-DFS,
Greedy Best-First, A*) are available through the GUI, along with
heuristic selection (Manhattan, Euclidean, Chebyshev) and movement
mode (orthogonal 4-dir or diagonal 8-dir).
"""
from maze_tkinter import main

if __name__ == "__main__":
    main()
