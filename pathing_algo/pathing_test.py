import numpy as np
from typing import Tuple, List
import heapq

def downsample_chunk(chunk: np.ndarray, robot_size: float, granularity: float) -> np.ndarray:
    """
    Downsample the chunk by taking max values in blocks.
    
    Parameters
    ----------
    chunk : numpy.ndarray
        Input occupancy grid
    robot_size : float
        Size of the robot (determines downsampling)
    granularity : float
        Grid cell size in meters
    
    Returns
    -------
    numpy.ndarray
        Downsampled occupancy grid
    """
    # Calculate block size based on robot size (1/4 of robot size)
    block_size = max(1, int(robot_size / (4 * granularity)))
    
    # Get chunk dimensions
    height, width = chunk.shape
    
    # Calculate new grid dimensions
    new_height = height // block_size
    new_width = width // block_size
    
    # Initialize downsampled chunk
    downsampled = np.zeros((new_height, new_width), dtype=chunk.dtype)
    
    # Downsample by taking max and checking for obstacles
    for y in range(new_height):
        for x in range(new_width):
            # Extract block
            block = chunk[
                y*block_size:(y+1)*block_size, 
                x*block_size:(x+1)*block_size
            ]
            
            # If any cell is an obstacle, mark as obstacle
            if np.any(block == 1):
                downsampled[y, x] = 1
            else:
                # Take max value of the block
                downsampled[y, x] = np.max(block)
    
    return downsampled

def generate_repulsion_field(chunk: np.ndarray, max_allowed_repulse: int, k_start: int = 1) -> np.ndarray:
    """
    Generate a repulsion field around obstacles.
    
    Parameters
    ----------
    chunk : numpy.ndarray
        Input occupancy grid
    max_allowed_repulse : int
        Maximum repulsion value
    k_start : int, optional
        Initial repulsion value
    
    Returns
    -------
    numpy.ndarray
        Repulsion-mapped grid
    """
    # Create a copy of the chunk to modify
    repulsion_map = chunk.copy()
    
    # Find obstacle locations
    obstacles = np.argwhere(chunk == 1)
    
    # 4-way directions (up, right, down, left)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Queue for BFS-style repulsion field generation
    queue = []
    visited = set()
    
    # Initialize queue with obstacles
    for obs in obstacles:
        queue.append((tuple(obs), k_start))
        visited.add(tuple(obs))
    
    # Generate repulsion field
    while queue:
        (y, x), k = queue.pop(0)
        
        # Stop if repulsion value becomes too low
        if k > max_allowed_repulse:
            continue
        
        # Update cell value
        repulsion_map[y, x] = max(repulsion_map[y, x], k)
        
        # Explore 4-way neighbors
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            # Check bounds
            if (0 <= ny < repulsion_map.shape[0] and 
                0 <= nx < repulsion_map.shape[1] and 
                (ny, nx) not in visited):
                
                # Add to queue with reduced repulsion value
                queue.append(((ny, nx), k + 1))
                visited.add((ny, nx))
    
    return repulsion_map

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Euclidean distance heuristic for A* pathfinding.
    
    Parameters
    ----------
    a : Tuple[int, int]
        Starting point
    b : Tuple[int, int]
        Goal point
    
    Returns
    -------
    float
        Euclidean distance between points
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar_navigation(
    repulsion_map: np.ndarray, 
    start: Tuple[int, int], 
    goal: Tuple[int, int], 
    max_allowed_repulse: int, 
    safe_distance: int
) -> List[Tuple[int, int]]:
    """
    A* pathfinding with obstacle avoidance and waypoint optimization.
    
    Parameters
    ----------
    repulsion_map : numpy.ndarray
        Repulsion-mapped grid
    start : Tuple[int, int]
        Starting grid coordinates
    goal : Tuple[int, int]
        Goal grid coordinates
    max_allowed_repulse : int
        Maximum allowed repulsion value
    safe_distance : int
        Distance threshold for waypoint optimization
    
    Returns
    -------
    List[Tuple[int, int]]
        Path from start to goal
    """
    # 4-way movement directions
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Priority queue for A*
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Tracking paths and costs
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        # Goal reached
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Explore neighbors
        for dy, dx in directions:
            neighbor = (current[0] + dy, current[1] + dx)
            
            # Check bounds
            if (0 <= neighbor[0] < repulsion_map.shape[0] and 
                0 <= neighbor[1] < repulsion_map.shape[1]):
                
                # Obstacle avoidance
                if repulsion_map[neighbor[0], neighbor[1]] >= max_allowed_repulse:
                    continue
                
                # Determine movement cost and heuristic
                if repulsion_map[current[0], current[1]] > safe_distance:
                    # Prioritize low repulsion cells
                    tentative_g = g_score[current] + repulsion_map[neighbor[0], neighbor[1]]
                else:
                    # Prioritize waypoint proximity
                    tentative_g = g_score[current] + heuristic(neighbor, goal)
                
                # Update path if better route found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return []

def navigate_chunk(
    chunk: np.ndarray,
    robot_size: float,
    prev_waypoint_pos: Tuple[int, int],
    new_waypoint_pos: Tuple[int, int],
    max_allowed_repulse: int,
    safe_distance: int,
    k_start: int = 1,
    granularity: float = 0.1
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Navigate through a chunk with obstacle avoidance and waypoint optimization.
    
    Parameters
    ----------
    chunk : numpy.ndarray
        Input occupancy grid chunk
    robot_size : float
        Size of the robot for downsampling
    prev_waypoint_pos : Tuple[int, int]
        Previous waypoint grid coordinates
    new_waypoint_pos : Tuple[int, int]
        New waypoint grid coordinates
    max_allowed_repulse : int
        Maximum repulsion value for obstacle avoidance
    safe_distance : int
        Distance threshold for waypoint optimization
    k_start : int, optional
        Initial repulsion value
    granularity : float, optional
        Grid cell size in meters
    
    Returns
    -------
    Tuple[numpy.ndarray, List[Tuple[int, int]]]
        Repulsion-mapped grid and path
    """
    # Downsample chunk
    downsampled_chunk = downsample_chunk(chunk, robot_size, granularity)
    
    # Generate repulsion field
    repulsion_map = generate_repulsion_field(
        downsampled_chunk, 
        max_allowed_repulse, 
        k_start
    )
    
    # Find path
    path = astar_navigation(
        repulsion_map, 
        prev_waypoint_pos, 
        new_waypoint_pos, 
        max_allowed_repulse, 
        safe_distance
    )
    
    return repulsion_map, path

# Example usage
if __name__ == "__main__":
    # Sample chunk with some obstacles
    chunk = np.array([
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Example parameters
    robot_size = 0.5  # 50 cm
    prev_waypoint = (0, 0)
    new_waypoint = (6, 7)
    max_allowed_repulse = 5
    safe_distance = 3
    
    # Navigate chunk
    repulsion_map, path = navigate_chunk(
        chunk, robot_size, prev_waypoint, new_waypoint, 
        max_allowed_repulse, safe_distance
    )
    
    # Debug print
    print("Repulsion Map:")
    print(repulsion_map)
    print("\nPath:", path)
