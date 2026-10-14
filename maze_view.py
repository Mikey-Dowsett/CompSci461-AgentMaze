"""Live-updating maze display built on matplotlib.

Two side-by-side panels (matplotlib has no native tabs):
  left  - grid view (white=free, black=wall)
  right - graph view (node per free cell, edge per open adjacency)

Usage:
    view = MazeView(maze)
    view.update(agent_pos)      # move agent marker, redraws
    view.show_path(path)        # animate a full path
    view.show()                 # keep window open (blocking)
"""
import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np


class MazeView:
    def __init__(self, maze, trail=True):
        self.maze = maze
        self.trail = trail
        self.agent_pos = maze.start
        self.visited = [maze.start]

        plt.ion()  # interactive mode so update() redraws immediately
        self.fig, (self.ax_maze, self.ax_graph) = plt.subplots(
            1, 2, figsize=(12, 6))
        self.fig.suptitle("Maze (left)  |  Graph representation (right)")

        # ---- left: grid view ----
        self.ax_maze.set_title("Grid")
        self.ax_maze.imshow(maze.grid, cmap="binary", origin="upper")
        self.ax_maze.set_xticks(np.arange(-0.5, maze.ncols, 1), minor=True)
        self.ax_maze.set_yticks(np.arange(-0.5, maze.nrows, 1), minor=True)
        self.ax_maze.grid(which="minor", color="gray", linewidth=0.5, alpha=0.5)
        self.ax_maze.set_xticks([])
        self.ax_maze.set_yticks([])

        sr, sc = maze.start
        gr, gc = maze.goal
        self.ax_maze.plot(sc, sr, "go", ms=12, label="start")  # x=col, y=row
        self.ax_maze.plot(gc, gr, "r*", ms=16, label="goal")
        # explored (search-frontier) shading, revealed progressively
        self.explored_scatter_maze = self.ax_maze.scatter(
            [], [], s=60, c="orange", alpha=0.55, zorder=1)
        (self.agent_dot,) = self.ax_maze.plot(sc, sr, "bo", ms=12, label="agent")
        (self.trail_line,) = self.ax_maze.plot([sc], [sr], "b-", lw=2, alpha=0.6)
        self.ax_maze.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),
                            ncol=4, fontsize="small")

        # ---- right: graph view ----
        self.ax_graph.set_title("Graph (node = free cell)")
        free = [(r, c) for r in range(maze.nrows)
                for c in range(maze.ncols) if maze.is_free((r, c))]
        segments = []
        for r, c in free:
            # only right/down to avoid duplicate edges
            for dr, dc in ((1, 0), (0, 1)):
                nb = (r + dr, c + dc)
                if maze.is_free(nb):
                    segments.append([(c, -r), (nb[1], -nb[0])])
        if segments:
            self.ax_graph.add_collection(
                LineCollection(segments, colors="gray", linewidths=1, alpha=0.6))
        if free:
            gx = [c for _, c in free]
            gy = [-r for r, _ in free]
            self.ax_graph.scatter(gx, gy, s=25, c="lightgray",
                                  edgecolors="dimgray", zorder=2)
        self.explored_scatter_graph = self.ax_graph.scatter(
            [], [], s=80, c="orange", alpha=0.55, zorder=3)
        # search-tree overlay: parent->child discovery edges + order labels
        self.tree_edges = []  # segments in graph coords, grown progressively
        self.tree_collection = LineCollection(
            [], colors="royalblue", linewidths=2, alpha=0.9, zorder=4)
        self.ax_graph.add_collection(self.tree_collection)
        self.tree_labels = []  # order-number annotations
        self.ax_graph.plot(sc, -sr, "go", ms=12, label="start")
        self.ax_graph.plot(gc, -gr, "r*", ms=16, label="goal")
        (self.graph_dot,) = self.ax_graph.plot(sc, -sr, "bo", ms=12, label="agent")
        (self.graph_trail,) = self.ax_graph.plot([sc], [-sr], "b-", lw=2, alpha=0.6)
        self.ax_graph.set_aspect("equal")
        self.ax_graph.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),
                             ncol=3, fontsize="small")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.05)

    def update(self, agent_pos, pause=0.2):
        """Move the agent marker to agent_pos (row, col) and redraw."""
        self.agent_pos = agent_pos
        self.visited.append(agent_pos)
        r, c = agent_pos
        self.agent_dot.set_data([c], [r])
        self.graph_dot.set_data([c], [-r])
        if self.trail:
            cols = [p[1] for p in self.visited]
            rows = [p[0] for p in self.visited]
            self.trail_line.set_data(cols, rows)
            self.graph_trail.set_data(cols, [-x for x in rows])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause)

    def reset(self, pause=0.2):
        """Teleport the agent back to start (clears the trail)."""
        self.agent_pos = self.maze.start
        self.visited = [self.maze.start]
        sr, sc = self.maze.start
        self.agent_dot.set_data([sc], [sr])
        self.graph_dot.set_data([sc], [-sr])
        self.trail_line.set_data([sc], [sr])
        self.graph_trail.set_data([sc], [-sr])
        self._set_explored([])
        self._clear_tree()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause)

    def _clear_tree(self):
        self.tree_edges = []
        self.tree_collection.set_segments([])
        for txt in self.tree_labels:
            txt.remove()
        self.tree_labels = []

    def _grow_tree(self, cell, order, parents):
        """Draw parent->child edge and order label for a newly explored cell."""
        r, c = cell
        parent = parents.get(cell)
        if parent is not None:
            pr, pc = parent
            self.tree_edges.append([(pc, -pr), (c, -r)])
            self.tree_collection.set_segments(self.tree_edges)
        self.tree_labels.append(self.ax_graph.text(
            c, -r, str(order), fontsize=7, ha="center", va="center",
            color="darkblue", fontweight="bold", zorder=5))

    def _set_explored(self, cells):
        if cells:
            cols = [c for _, c in cells]
            rows = [r for r, _ in cells]
            self.explored_scatter_maze.set_offsets(list(zip(cols, rows)))
            self.explored_scatter_graph.set_offsets(
                list(zip(cols, [-r for r in rows])))
        else:
            self.explored_scatter_maze.set_offsets(np.empty((0, 2)))
            self.explored_scatter_graph.set_offsets(np.empty((0, 2)))

    def show_search(self, explored, walk, parents=None, pause=0.05,
                    walk_pause=0.2, loop=False, loop_pause=1.0):
        """Animate the agent's actual search.

        Phase 1: explored cells light up orange in expansion order while the
        discovery tree grows on the graph panel (blue parent->child edges,
        each node labelled with its search order number).
        Phase 2: the agent physically walks (including backtracking for DFS).
        """
        parents = parents or {}
        try:
            while True:
                shown = []
                for order, cell in enumerate(explored):
                    if not plt.fignum_exists(self.fig.number):
                        return
                    shown.append(cell)
                    self._set_explored(shown)
                    self._grow_tree(cell, order, parents)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    plt.pause(pause)
                first = True
                for pos in walk:
                    if first and pos == self.agent_pos and len(self.visited) == 1:
                        first = False
                        continue
                    first = False
                    if not plt.fignum_exists(self.fig.number):
                        return
                    self.update(pos, pause=walk_pause)
                if not loop:
                    return
                plt.pause(loop_pause)
                if not plt.fignum_exists(self.fig.number):
                    return
                self.reset(pause=walk_pause)
        except KeyboardInterrupt:
            return

    def show_path(self, path, pause=0.2, loop=False, loop_pause=1.0):
        """Animate the agent walking along a list of (row, col) positions.

        If loop=True, replay forever until the window is closed
        (or Ctrl+C in the terminal).
        """
        for pos in path:
            if not self.maze.is_free(pos):
                raise ValueError(f"Path steps into wall/out of bounds: {pos}")
        try:
            while True:
                first = True
                for pos in path:
                    if first and pos == self.agent_pos and len(self.visited) == 1:
                        first = False
                        continue  # skip re-drawing the starting cell
                    first = False
                    if not plt.fignum_exists(self.fig.number):
                        return
                    self.update(pos, pause=pause)
                if not loop:
                    return
                plt.pause(loop_pause)
                if not plt.fignum_exists(self.fig.number):
                    return
                self.reset(pause=pause)
        except KeyboardInterrupt:
            return

    def show(self):
        """Block and keep the window open."""
        plt.ioff()
        plt.show()
