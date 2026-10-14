"""Tkinter UI wrapper for the maze project with Canvas visualization."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from maze import Maze
from agent import bfs_search, dfs_search


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

        # Config defaults
        self.nrows_var = tk.IntVar(value=15)
        self.ncols_var = tk.IntVar(value=20)
        self.wall_prob_var = tk.DoubleVar(value=0.25)
        self.start_r_var = tk.IntVar(value=0)
        self.start_c_var = tk.IntVar(value=0)
        self.goal_r_var = tk.IntVar(value=14)
        self.goal_c_var = tk.IntVar(value=19)
        self.random_placement = tk.BooleanVar(value=False)
        self.algo_var = tk.StringVar(value="bfs")

        self.build_ui()

    def build_ui(self):
        # Main layout: left config, right canvas
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- Left config panel ----
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Rows / Cols
        ttk.Label(config_frame, text="Rows:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(config_frame, from_=5, to=50, textvariable=self.nrows_var, width=8).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text="Cols:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(config_frame, from_=5, to=50, textvariable=self.ncols_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Wall probability
        ttk.Label(config_frame, text="Wall prob:").grid(row=2, column=0, sticky=tk.W)
        ttk.Scale(config_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, variable=self.wall_prob_var, length=120).grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(config_frame, textvariable=self.wall_prob_var).grid(row=2, column=2, sticky=tk.W)

        # Start / Goal
        ttk.Label(config_frame, text="Start (r c):").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.start_r_var, width=4).grid(row=3, column=1, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.start_c_var, width=4).grid(row=3, column=2, sticky=tk.W)

        ttk.Label(config_frame, text="Goal (r c):").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.goal_r_var, width=4).grid(row=4, column=1, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.goal_c_var, width=4).grid(row=4, column=2, sticky=tk.W)

        # Random placement toggle
        ttk.Checkbutton(config_frame, text="Random start/goal", variable=self.random_placement).grid(row=4, column=2, sticky=tk.W, pady=2)

        # Algorithm
        ttk.Label(config_frame, text="Algorithm:").grid(row=5, column=0, sticky=tk.W, pady=(8, 0))
        algo_combo = ttk.Combobox(config_frame, textvariable=self.algo_var, values=["bfs", "dfs"], state="readonly", width=8)
        algo_combo.grid(row=5, column=1, sticky=tk.W, pady=(8, 0))

        # Animation mode
        ttk.Label(config_frame, text="Animation:").grid(row=6, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Radiobutton(config_frame, text="Instant", variable=self.animation_mode, value="instant").grid(row=6, column=1, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="Step-by-step", variable=self.animation_mode, value="step").grid(row=7, column=1, sticky=tk.W)

        # Buttons
        btn_frame = ttk.Frame(config_frame)
        btn_frame.grid(row=8, column=0, columnspan=3, pady=15)
        ttk.Button(btn_frame, text="Generate Maze", command=self.generate_maze).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Run Search", command=self.run_search).pack(side=tk.LEFT, padx=3)

        # Status
        self.status_var = tk.StringVar(value="Ready. Click 'Generate Maze' to start.")
        ttk.Label(config_frame, textvariable=self.status_var, wraplength=220).grid(row=9, column=0, columnspan=3, pady=10)

        # ---- Right canvas panel ----
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, highlightbackground="gray", width=500, height=500)
        self.canvas.pack(padx=10, pady=10)

        # Bind resize to redraw
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())

    def generate_maze(self):
        try:
            nrows = self.nrows_var.get()
            ncols = self.ncols_var.get()
            wall_prob = self.wall_prob_var.get()
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

            self.maze, _ = Maze.generate_solvable(
                nrows, ncols, wall_prob,
                start=start, goal=goal
            )
            self.result = None
            self.animation_running = False
            self.step_index = 0
            self.redraw_canvas()
            self.status_var.set(f"Maze ready: {nrows}x{ncols}, start={start}, goal={goal}. Click 'Run Search'.")
        except Exception as e:
            messagebox.showerror("Maze Generation Error", str(e))
            self.status_var.set("Generation failed.")

    def run_search(self):
        if self.maze is None:
            messagebox.showwarning("No Maze", "Generate a maze first.")
            return
        if self.animation_running:
            return

        algo = self.algo_var.get()
        search_fn = bfs_search if algo == "bfs" else dfs_search

        try:
            self.result = search_fn(self.maze)
            self.animation_running = True
            self.step_index = 0

            mode = self.animation_mode.get()
            if mode == "instant":
                self.run_instant()
            else:
                self.run_step_by_step()
        except Exception as e:
            messagebox.showerror("Search Error", str(e))
            self.animation_running = False

    def run_instant(self):
        # Draw full explored set and reconstructed path immediately
        explored = self.result.explored
        path = self.result.path or []
        self.draw_explored(explored)
        self.draw_path(path)
        self.animation_running = False
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
            self.animation_after_id = self.root.after(80, lambda: self.animate_next_step(explored, path))
            return

        # Phase 2: animate reconstructed path (fastest route found)
        if self.step_index == len(explored):
            # Start walking the reconstructed path
            self.walk_index = 0
            self.current_path = [path[0]] if path else []
            self.step_index += 1  # mark phase transition
            self.animation_after_id = self.root.after(150, lambda: self.animate_path_step(path))
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
            self.animation_after_id = self.root.after(120, lambda: self.animate_path_step(path))
        else:
            self.animation_running = False
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

        # Draw start and goal
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
