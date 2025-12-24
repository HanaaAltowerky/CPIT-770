import heapq
from typing import List, Tuple, Set, Optional

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


# Example usage
if __name__ == "__main__":
    # 0 = walkable, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (3, 3)
    
    path = astar(grid, start, goal)
    
    if path:
        print("\n\nPath found:")
        for pos in path:
            print(f"  {pos}")
        
        # Visualize
        print("\nVisualization (S=start, G=goal, *=path, #=obstacle):")
        for r in range(len(grid)):
            row = ""
            for c in range(len(grid[0])):
                if (r, c) == start:
                    row += "S "
                elif (r, c) == goal:
                    row += "G "
                elif (r, c) in path:
                    row += "* "
                elif grid[r][c] == 1:
                    row += "# "
                else:
                    row += ". "
            print(row)
    else:
        print("\nNo path found!")