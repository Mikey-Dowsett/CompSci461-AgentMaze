"""Tkinter UI wrapper for the maze project with Canvas visualization."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from maze import Maze
from agent import bfs_search, dfs_search, id_dfs_search, best_first_search, astar_search, HEURISTICS


class MazeTkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Explorer")
        self.root.geometry("900x650")
        self.root.minsize(700, 500)

        # State
        self.maze = None
        self.result = None
        self.animation_running = False
        self.animation_after_id = None
        self.step_index = 0
        self.offset_x = 0
        self.offset_y = 0
        self.animation_mode = tk.StringVar(value="instant")  # "step" or "instant"
        self.delay_var = tk.DoubleVar(value=50.0)  # Animation delay in ms (1-100, maps to 10ms-0.01ms)

        # Config defaults
        self.nrows_var = tk.IntVar(value=15)
        self.ncols_var = tk.IntVar(value=15)
        self.wall_prob_var = tk.DoubleVar(value=0.25)
        self.wall_prob_display = tk.StringVar(value=f"{self.wall_prob_var.get():.2f}")
        self.start_r_var = tk.IntVar(value=0)
        self.start_c_var = tk.IntVar(value=0)
        self.goal_r_var = tk.IntVar(value=14)
        self.goal_c_var = tk.IntVar(value=14)
        self.random_placement = tk.BooleanVar(value=False)
        self.algo_var = tk.StringVar(value="bfs")
        self.heuristic_var = tk.StringVar(value="manhattan")
        self.movement_var = tk.StringVar(value="orthogonal")
        self.seed_var = tk.StringVar(value="70804")  # Empty = random seed

        self.build_ui()

    def build_ui(self):
        # Main layout: left config, right canvas
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- Left config panel ----
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Rows / Cols
        ttk.Label(config_frame, text="Rows:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(config_frame, from_=5, to=50, textvariable=self.nrows_var, width=8).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(config_frame, text="Cols:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(config_frame, from_=5, to=50, textvariable=self.ncols_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=5)

        # Wall probability
        ttk.Label(config_frame, text="Wall prob:").grid(row=2, column=0, sticky=tk.W, pady=5)
        tk.Scale(config_frame, from_=0.1, to=0.7, orient=tk.HORIZONTAL, variable=self.wall_prob_var, length=120, resolution=0.05, command=lambda v: self.wall_prob_display.set(f"{float(v):.2f}")).grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(config_frame, textvariable=self.wall_prob_display).grid(row=2, column=2, sticky=tk.W, pady=5)

        # Start / Goal
        ttk.Label(config_frame, text="Start (r c):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.start_r_var, width=4).grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.start_c_var, width=4).grid(row=3, column=2, sticky=tk.W, pady=5)

        ttk.Label(config_frame, text="Goal (r c):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.goal_r_var, width=4).grid(row=4, column=1, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.goal_c_var, width=4).grid(row=4, column=2, sticky=tk.W, pady=5)

        # Random placement toggle
        ttk.Checkbutton(config_frame, text="Random start/goal", variable=self.random_placement).grid(row=4, column=2, sticky=tk.W, pady=5)

        # Algorithm
        ttk.Label(config_frame, text="Algorithm:").grid(row=5, column=0, sticky=tk.W, pady=5)
        algo_combo = ttk.Combobox(config_frame, textvariable=self.algo_var, values=["bfs", "dfs", "iddfs", "best_first", "astar"], state="readonly", width=12)
        algo_combo.grid(row=5, column=1, sticky=tk.W, pady=5)

        # Heuristic
        ttk.Label(config_frame, text='Heuristic:').grid(row=6, column=0, sticky=tk.W, pady=5)
        heuristic_combo = ttk.Combobox(config_frame, textvariable=self.heuristic_var, values=list(HEURISTICS.keys()), state='readonly', width=12)
        heuristic_combo.grid(row=6, column=1, sticky=tk.W, pady=5)

        # Movement mode
        ttk.Label(config_frame, text='Movement:').grid(row=7, column=0, sticky=tk.W, pady=5)
        move_frame = ttk.Frame(config_frame)
        move_frame.grid(row=7, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(move_frame, text='Orthogonal', variable=self.movement_var, value='orthogonal').pack(side=tk.LEFT)
        ttk.Radiobutton(move_frame, text='Diagonal', variable=self.movement_var, value='diagonal').pack(side=tk.LEFT)

        # Seed
        ttk.Label(config_frame, text="Seed (empty = random):").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.seed_var, width=12).grid(row=8, column=1, sticky=tk.W, pady=5)

        # Animation mode
        ttk.Label(config_frame, text="Animation:").grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(config_frame, text="Instant", variable=self.animation_mode, value="instant").grid(row=9, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(config_frame, text="Step-by-step", variable=self.animation_mode, value="step").grid(row=10, column=1, sticky=tk.W, pady=5)

        # Animation delay slider (1-100, maps to 10ms-0.01ms)
        ttk.Label(config_frame, text="Speed (1-100):").grid(row=11, column=0, sticky=tk.W, pady=5)
        self.delay_var = tk.DoubleVar(value=50.0)
        tk.Scale(config_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.delay_var, length=100, resolution=1).grid(row=11, column=1, sticky=tk.W, pady=5)
        self.delay_display = tk.StringVar(value="50")
        ttk.Label(config_frame, textvariable=self.delay_display).grid(row=11, column=2, sticky=tk.W, pady=5)

        def update_delay_display(*args):
            # Map 1-100 to delay: 10ms (slow) to 0.01ms (fast)
            delay = 10.0 / self.delay_var.get()
            self.delay_display.set(f"{delay:.2f} ms")
        self.delay_var.trace_add("write", update_delay_display)

        # Buttons
        btn_frame = ttk.Frame(config_frame)
        btn_frame.grid(row=12, column=0, columnspan=3, pady=15)
        ttk.Button(btn_frame, text="Generate Maze", command=self.generate_maze).pack(side=tk.LEFT, padx=3)
        self.run_btn = ttk.Button(btn_frame, text="Run Search", command=self.run_search)
        self.run_btn.pack(side=tk.LEFT, padx=3)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_animation, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=3)

        # Status
        self.status_var = tk.StringVar(value="Ready. Click 'Generate Maze' to start.")
        ttk.Label(config_frame, textvariable=self.status_var, wraplength=220).grid(row=13, column=0, columnspan=3, pady=10)

        # ---- Right display panel with tabs ----
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Maze tab
        maze_tab = ttk.Frame(self.notebook)
        self.notebook.add(maze_tab, text="Maze")
        self.canvas = tk.Canvas(maze_tab, bg="white", highlightthickness=1, highlightbackground="gray", width=500, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())

        # Graph tab (matplotlib embedded)
        graph_tab = ttk.Frame(self.notebook)
        self.notebook.add(graph_tab, text="Graph")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.graph_fig = Figure(figsize=(6, 5), dpi=100)
        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=graph_tab)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def generate_maze(self):
        try:
            nrows = self.nrows_var.get()
            ncols = self.ncols_var.get()
            wall_prob = round(float(self.wall_prob_var.get()), 2)
            if self.random_placement.get():
                rng = np.random.default_rng()
                start = (int(rng.integers(nrows)), int(rng.integers(ncols)))
                goal = (int(rng.integers(nrows)), int(rng.integers(ncols)))
                while goal == start:
                    goal = (int(rng.integers(nrows)), int(rng.integers(ncols)))
                self.start_r_var.set(start[0])
                self.start_c_var.set(start[1])
                self.goal_r_var.set(goal[0])
                self.goal_c_var.set(goal[1])
            else:
                start = (self.start_r_var.get(), self.start_c_var.get())
                goal = (self.goal_r_var.get(), self.goal_c_var.get())

            # Validate
            if not (0 <= start[0] < nrows and 0 <= start[1] < ncols):
                messagebox.showerror("Invalid Start", f"Start must be within 0..{nrows-1}, 0..{ncols-1}")
                return
            if not (0 <= goal[0] < nrows and 0 <= goal[1] < ncols):
                messagebox.showerror("Invalid Goal", f"Goal must be within 0..{nrows-1}, 0..{ncols-1}")
                return
            if start == goal:
                messagebox.showwarning("Same Position", "Start and goal are the same; adjusting goal to (0,0).")
                goal = (0, 0)
                self.goal_r_var.set(0)
                self.goal_c_var.set(0)

            self.status_var.set(f"Generating {nrows}x{ncols} maze (walls~{wall_prob})...")
            self.root.update()

            # Parse seed - empty string means use random seed
            seed = None
            seed_str = self.seed_var.get().strip()
            if seed_str:
                try:
                    seed = int(seed_str)
                except ValueError:
                    messagebox.showerror("Invalid Seed", f"Seed must be an integer, got: {seed_str}")
                    return

            self.maze, _ = Maze.generate_solvable(
                nrows, ncols, wall_prob,
                start=start, goal=goal, seed=seed
            )
            self.result = None
            self.animation_running = False
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.step_index = 0
            self.redraw_canvas()
            self.status_var.set(f"Maze ready: {nrows}x{ncols}, start={start}, goal={goal}. Click 'Run Search'.")
        except Exception as e:
            messagebox.showerror("Maze Generation Error", str(e))
            self.status_var.set("Generation failed.")

    def stop_animation(self):
        self.animation_running = False
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        self.stop_btn.config(state="disabled")
        self.status_var.set("Search stopped.")


    def run_search(self):
        if self.maze is None:
            messagebox.showwarning("No Maze", "Generate a maze first.")
            return
        if self.animation_running:
            return

        algo = self.algo_var.get()
        heuristic = self.heuristic_var.get()
        diagonal = self.movement_var.get() == "diagonal"

        if algo == "bfs":
            search_fn = lambda m: bfs_search(m)
        elif algo == "dfs":
            search_fn = lambda m: dfs_search(m)
        elif algo == "iddfs":
            search_fn = lambda m: id_dfs_search(m, heuristic=heuristic, diagonal=diagonal)
        elif algo == "best_first":
            search_fn = lambda m: best_first_search(m, heuristic=heuristic, diagonal=diagonal)
        elif algo == "astar":
            search_fn = lambda m: astar_search(m, heuristic=heuristic, diagonal=diagonal)
        else:
            search_fn = bfs_search

        try:
            self.result = search_fn(self.maze)
            self.animation_running = True
            self.run_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.step_index = 0
            # Store parents for graph tree edges
            self.parents = getattr(self.result, 'parents', {})

            mode = self.animation_mode.get()
            if mode == "instant":
                self.current_explored = []
                self.current_path = []
                self.run_instant()
            else:
                self.run_step_by_step()
        except Exception as e:
            messagebox.showerror("Search Error", str(e))
            self.animation_running = False
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def run_instant(self):
        # Clear previous animation state and redraw base
        self.current_explored = []
        self.current_path = []
        self.redraw_canvas()
        # Draw full explored set and reconstructed path immediately
        explored = self.result.explored
        path = self.result.path or []
        self.current_explored = list(explored)
        self.current_path = list(path)
        self.draw_explored(explored)
        self.draw_path(path)
        self.animation_running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        explored_len = len(explored)
        path_len = len(path) - 1 if path else 0
        self.status_var.set(
            f"{self.algo_var.get().upper()}: explored {explored_len} cells, "
            f"path length {path_len} steps."
        )

    def run_step_by_step(self):
        # Animate step by step using after()
        explored = self.result.explored
        path = self.result.path or []
        self.step_index = 0
        self.current_explored = []
        self.current_path = []
        self.redraw_canvas()
        self.animate_next_step(explored, path)

    def animate_next_step(self, explored, path):
        if not self.animation_running:
            return
        # Phase 1: reveal explored cells
        if self.step_index < len(explored):
            cell = explored[self.step_index]
            self.current_explored.append(cell)
            self.step_index += 1
            self.redraw_canvas()
            self.draw_explored(self.current_explored)
            # Small delay
            delay = max(1, int(10.0 / self.delay_var.get()))
            self.animation_after_id = self.root.after(delay, lambda: self.animate_next_step(explored, path))
            return

        # Phase 2: animate reconstructed path (fastest route found)
        if self.step_index == len(explored):
            # Start walking the reconstructed path
            self.walk_index = 0
            self.current_path = [path[0]] if path else []
            self.step_index += 1  # mark phase transition
            delay = max(1, int(10.0 / self.delay_var.get()))
            self.animation_after_id = self.root.after(delay, lambda: self.animate_path_step(path))
            return

    def animate_path_step(self, path):
        if not self.animation_running:
            return
        if self.walk_index < len(path):
            pos = path[self.walk_index]
            self.current_path.append(pos)
            self.walk_index += 1
            self.redraw_canvas()
            self.draw_explored(getattr(self, "current_explored", []))
            self.draw_path(self.current_path)
            delay = max(1, int(10.0 / self.delay_var.get()))
            self.animation_after_id = self.root.after(delay, lambda: self.animate_path_step(path))
        else:
            self.animation_running = False
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            explored_len = len(getattr(self, "current_explored", []))
            path_len = len(self.current_path) - 1 if self.current_path else 0
            self.status_var.set(
                f"{self.algo_var.get().upper()}: explored {explored_len} cells, "
                f"path length {path_len} steps."
            )

    def draw_explored(self, explored_set):
        if self.maze is None:
            return
        cell_w = self.cell_w
        cell_h = self.cell_h
        for (r, c) in explored_set:
            x1 = self.offset_x + c * cell_w + 1
            y1 = self.offset_y + r * cell_h + 1
            x2 = self.offset_x + (c + 1) * cell_w - 1
            y2 = self.offset_y + (r + 1) * cell_h - 1
            self.canvas.create_oval(x1, y1, x2, y2, fill="orange", outline="", tags="explored")
        self.canvas.tag_raise("start_goal")
        self.redraw_graph()

    def draw_path(self, path):
        if self.maze is None or not path:
            return
        cell_w = self.cell_w
        cell_h = self.cell_h
        # Clear previous path lines
        self.canvas.delete("path_line")
        # Draw line segments connecting path points
        for i in range(1, len(path)):
            r1, c1 = path[i - 1]
            r2, c2 = path[i]
            x1 = self.offset_x + c1 * cell_w + cell_w / 2
            y1 = self.offset_y + r1 * cell_h + cell_h / 2
            x2 = self.offset_x + c2 * cell_w + cell_w / 2
            y2 = self.offset_y + r2 * cell_h + cell_h / 2
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=3, tags="path_line", capstyle=tk.ROUND)
        # Draw agent dot at last position
        self.canvas.delete("agent")
        if path:
            r, c = path[-1]
            cx = self.offset_x + c * cell_w + cell_w / 2
            cy = self.offset_y + r * cell_h + cell_h / 2
            self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="blue", outline="", tags="agent")
        self.canvas.tag_raise("start_goal")
        self.redraw_graph()

    def redraw_canvas(self):
        if self.maze is None:
            return
        # Clear previous drawings except base grid
        self.canvas.delete("cell")
        self.canvas.delete("explored")
        self.canvas.delete("path_line")
        self.canvas.delete("agent")
        self.canvas.delete("start_goal")

        # Calculate cell size based on canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
        nrows = self.maze.nrows
        ncols = self.maze.ncols
        # Square cells: use same size for width and height
        cell_size = min(canvas_w // ncols, canvas_h // nrows)
        cell_size = max(10, cell_size)
        self.cell_w = cell_size
        self.cell_h = cell_size

        # Center the grid in the square canvas
        self.offset_x = (canvas_w - ncols * self.cell_w) // 2
        self.offset_y = (canvas_h - nrows * self.cell_h) // 2
        # Draw grid
        for r in range(nrows):
            for c in range(ncols):
                x1 = self.offset_x + c * self.cell_w
                y1 = self.offset_y + r * self.cell_h
                x2 = x1 + self.cell_w
                y2 = y1 + self.cell_h
                if self.maze.grid[r, c] == 1:
                    fill = "black"
                else:
                    fill = "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="gray", tags="cell")

        # Redraw any active animation state
        if self.result is not None:
            explored = getattr(self, "current_explored", self.result.explored)
            self.draw_explored(explored)
            path = getattr(self, "current_path", self.result.path or [])
            if path:
                self.draw_path(path)
            else:
                # If no path yet, just show start
                self.draw_path([self.maze.start])

        # Draw start and goal on top so they don't vanish under orange dots
        sr, sc = self.maze.start
        gr, gc = self.maze.goal
        # Start (green circle)
        cx_s = self.offset_x + sc * self.cell_w + self.cell_w / 2
        cy_s = self.offset_y + sr * self.cell_h + self.cell_h / 2
        self.canvas.create_oval(cx_s - 8, cy_s - 8, cx_s + 8, cy_s + 8, fill="green", outline="darkgreen", width=2, tags="start_goal")
        # Goal (red star approximation: diamond)
        cx_g = self.offset_x + gc * self.cell_w + self.cell_w / 2
        cy_g = self.offset_y + gr * self.cell_h + self.cell_h / 2
        self.canvas.create_polygon(cx_g, cy_g - 10, cx_g + 10, cy_g, cx_g, cy_g + 10, cx_g - 10, cy_g, fill="red", outline="darkred", tags="start_goal")
        self.canvas.tag_raise("start_goal")
        self.redraw_graph()

    def redraw_graph(self):
        if self.maze is None or not hasattr(self, 'graph_ax'):
            return
        self.graph_ax.clear()
        self.graph_ax.set_title("Graph (node = free cell)")
        # Build free cells and edges
        free = [(r, c) for r in range(self.maze.nrows)
                for c in range(self.maze.ncols) if self.maze.is_free((r, c))]
        segments = []
        for r, c in free:
            for dr, dc in ((1, 0), (0, 1)):
                nb = (r + dr, c + dc)
                if self.maze.is_free(nb):
                    segments.append([(c, -r), (nb[1], -nb[0])])
        if segments:
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, colors="gray", linewidths=1, alpha=0.6)
            self.graph_ax.add_collection(lc)
        if free:
            gx = [c for _, c in free]
            gy = [-r for r, _ in free]
            self.graph_ax.scatter(gx, gy, s=25, c="lightgray",
                                  edgecolors="dimgray", zorder=2)
        # Start / goal
        sr, sc = self.maze.start
        gr, gc = self.maze.goal
        self.graph_ax.plot(sc, -sr, "go", ms=12, label="start")
        self.graph_ax.plot(gc, -gr, "r*", ms=16, label="goal")
        # Active state
        if self.result is not None:
            explored = getattr(self, "current_explored", self.result.explored)
            path = getattr(self, "current_path", self.result.path or [])
            # Search tree edges (parent -> child) in orange
            parents = getattr(self, 'parents', {})
            if parents and explored:
                tree_segments = []
                for child in explored:
                    parent = parents.get(child)
                    if parent is not None:
                        pr, pc = parent
                        r, c = child
                        tree_segments.append([(pc, -pr), (c, -r)])
                if tree_segments:
                    from matplotlib.collections import LineCollection
                    lc = LineCollection(tree_segments, colors="orange", linewidths=2, alpha=0.9)
                    self.graph_ax.add_collection(lc)
            if explored:
                ex = [c for _, c in explored]
                ey = [-r for r, _ in explored]
                self.graph_ax.scatter(ex, ey, s=80, c="orange", alpha=0.55, zorder=3)
            if path:
                px = [c for _, c in path]
                py = [-r for r, _ in path]
                self.graph_ax.plot(px, py, "b-", lw=2, alpha=0.6, zorder=4)
                if path:
                    r, c = path[-1]
                    self.graph_ax.plot(c, -r, "bo", ms=12, zorder=5)
        self.graph_ax.set_aspect("equal")
        self.graph_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),
                             ncol=3, fontsize="small")
        self.graph_fig.tight_layout()
        self.graph_canvas.draw()

    def on_closing(self):
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MazeTkApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
