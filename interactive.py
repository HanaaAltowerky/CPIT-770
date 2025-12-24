import heapq
from typing import List, Tuple, Set, Optional
import tkinter as tk
from tkinter import messagebox

def astar(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm.

    Args:
        grid: 2D grid where 0 = walkable, 1 = obstacle
        start: Starting position (row, col)
        goal: Goal position (row, col)

    Returns:
        List of positions from start to goal, or None if no path exists
    """
    rows, cols = len(grid), len(grid[0])

    def h(pos: Tuple[int, int]) -> float:
        """Heuristic: Manhattan distance"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        r, c = pos
        candidates = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        return [(nr, nc) for nr, nc in candidates
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0]

    # Priority queue: (f_score, counter, position)
    counter = 0
    pq = [(h(start), counter, start)]
    counter += 1

    came_from = {}
    g_score = {start: 0}
    visited: Set[Tuple[int, int]] = set()

    while pq:
        _, _, curr = heapq.heappop(pq)

        if curr == goal:
            # Reconstruct path
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            return path[::-1]

        if curr in visited:
            continue
        visited.add(curr)

        for nb in neighbors(curr):
            tentative_g = g_score[curr] + 1

            if nb not in g_score or tentative_g < g_score[nb]:
                came_from[nb] = curr
                g_score[nb] = tentative_g
                f_score = tentative_g + h(nb)
                heapq.heappush(pq, (f_score, counter, nb))
                counter += 1

    return None  # No path found


# ---------------- GUI (Canvas grid) ----------------
class AStarGUI:
    def __init__(self, root, rows=12, cols=12, cell=42, pad=6):
        self.root = root
        self.root.title("A* Search - Canvas Grid")

        self.rows = rows
        self.cols = cols
        self.cell = cell
        self.pad = pad

        # 0 walkable, 1 obstacle
        self.grid: List[List[int]] = [[0 for _ in range(cols)] for _ in range(rows)]

        self.mode = tk.StringVar(value="wall")  # wall / start / goal
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.path: List[Tuple[int, int]] = []

        # palette: d9ead3,d0e2f4,f3f3f3,d9d3e9,1c1c1c
        self.C_PATH = "#d9ead3"
        self.C_START = "#d0e2f4"
        self.C_EMPTY = "#f3f3f3"
        self.C_GOAL = "#d9d3e9"
        self.C_WALL = "#1c1c1c"

        controls = tk.Frame(root, padx=10, pady=10)
        controls.pack(fill="x")

        tk.Label(controls, text="Click mode:").pack(side="left")
        tk.Radiobutton(controls, text="Walls", variable=self.mode, value="wall").pack(side="left", padx=6)
        tk.Radiobutton(controls, text="Start", variable=self.mode, value="start").pack(side="left", padx=6)
        tk.Radiobutton(controls, text="Goal",  variable=self.mode, value="goal").pack(side="left", padx=6)

        tk.Button(controls, text="Run A*", command=self.run).pack(side="left", padx=12)
        tk.Button(controls, text="Clear Path", command=self.clear_path).pack(side="left", padx=4)
        tk.Button(controls, text="Reset All", command=self.reset_all).pack(side="left", padx=4)

        w_px = self.cols * self.cell + 2 * self.pad
        h_px = self.rows * self.cell + 2 * self.pad
        self.canvas = tk.Canvas(root, width=w_px, height=h_px, highlightthickness=0, bg="white")
        self.canvas.pack(padx=10, pady=10)

        self.rects = {}   # (r,c) -> rect_id
        self.labels = {}  # (r,c) -> text_id

        for r in range(self.rows):
            for c in range(self.cols):
                x0 = self.pad + c * self.cell
                y0 = self.pad + r * self.cell
                x1 = x0 + self.cell - 2
                y1 = y0 + self.cell - 2

                rect = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=self.C_EMPTY,
                    outline="#cccccc",
                    width=1
                )
                text = self.canvas.create_text(
                    (x0 + x1) / 2, (y0 + y1) / 2,
                    text="",
                    font=("Helvetica", 16, "bold"),
                    fill="black"
                )

                self.rects[(r, c)] = rect
                self.labels[(r, c)] = text

        self.canvas.bind("<Button-1>", self.on_click)
        self.refresh_all()

    def pixel_to_cell(self, px, py):
        px -= self.pad
        py -= self.pad
        if px < 0 or py < 0:
            return None
        c = px // self.cell
        r = py // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (int(r), int(c))
        return None

    def on_click(self, event):
        pos = self.pixel_to_cell(event.x, event.y)
        if pos is None:
            return

        r, c = pos
        mode = self.mode.get()

        # clear only the path display
        self.path = []

        if mode == "wall":
            if self.start == (r, c) or self.goal == (r, c):
                return
            self.grid[r][c] = 0 if self.grid[r][c] == 1 else 1

        elif mode == "start":
            if self.grid[r][c] == 1 or self.goal == (r, c):
                return
            self.start = (r, c)

        elif mode == "goal":
            if self.grid[r][c] == 1 or self.start == (r, c):
                return
            self.goal = (r, c)

        self.refresh_all()

    def run(self):
        self.path = []
        self.refresh_all()

        if self.start is None or self.goal is None:
            messagebox.showwarning("Missing", "Please set both Start and Goal.")
            return

        path = astar(self.grid, self.start, self.goal)
        if not path:
            messagebox.showinfo("No path", "No path found.")
            return

        self.path = path
        self.refresh_all()

    def clear_path(self):
        self.path = []
        self.refresh_all()

    def reset_all(self):
        self.start = None
        self.goal = None
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.path = []
        self.refresh_all()

    def refresh_all(self):
        path_set = set(self.path)

        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.rects[(r, c)]
                label = self.labels[(r, c)]

                if self.start == (r, c):
                    fill, text, fg = self.C_START, "S", "black"
                elif self.goal == (r, c):
                    fill, text, fg = self.C_GOAL, "G", "black"
                elif self.grid[r][c] == 1:
                    fill, text, fg = self.C_WALL, "W", "white"
                elif (r, c) in path_set and (r, c) != self.start and (r, c) != self.goal:
                    fill, text, fg = self.C_PATH, "•", "black"
                else:
                    fill, text, fg = self.C_EMPTY, "", "black"

                self.canvas.itemconfig(rect, fill=fill)
                self.canvas.itemconfig(label, text=text, fill=fg)


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    AStarGUI(root, rows=12, cols=12)
    root.mainloop()
