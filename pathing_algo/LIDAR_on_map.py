import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from LIDAR import LidarConfig, Obstacle, simulate_lidar_scan
from global_mapping import WaypointMaps, MapConfig, initialize_waypoint_maps


def update_maps_with_lidar(
    maps: WaypointMaps,
    lidar_scan: np.ndarray,
    robot_pose: Tuple[float, float, float],  # (x, y, theta) in meters and radians
    lidar_config: LidarConfig,
    temp_obstacle_value: int = 2  # Using 2 to distinguish temporary from permanent obstacles
) -> WaypointMaps:
    """
    Update the waypoint maps with temporary obstacles detected by LIDAR.
    
    Parameters
    ----------
    maps : WaypointMaps
        The waypoint maps data structure to update
    lidar_scan : numpy.ndarray
        Shape (N, 2) array of [azimuth (rad), range (m)] LIDAR readings
    robot_pose : Tuple[float, float, float]
        Robot's current pose as (x, y, theta) where:
        - x, y are in meters
        - theta is in radians (0 is along positive x-axis)
    lidar_config : LidarConfig
        LIDAR configuration parameters
    temp_obstacle_value : int, optional
        Value to use for temporary obstacles in the occupancy grid
        
    Returns
    -------
    WaypointMaps
        Updated maps with temporary obstacles
    
    Notes
    -----
    Map cell values:
        -1: Unknown
         0: Free space
         1: Permanent obstacle
         2: Temporary obstacle (from LIDAR)
    """
    # Create a copy of the maps to avoid modifying the original
    updated_maps = WaypointMaps(
        occupancy_maps=maps.occupancy_maps.copy(),
        waypoint_indices=maps.waypoint_indices.copy(),
        config=maps.config
    )
    
    robot_x, robot_y, robot_theta = robot_pose
    
    # Find which waypoint maps need to be updated based on robot position
    relevant_map_indices = []
    for i in range(maps.config.num_waypoints):
        min_x, max_x, min_y, max_y = maps.get_map_boundaries_meters(i)
        # Add some margin based on LIDAR range
        margin = lidar_config.max_range
        if (min_x - margin <= robot_x <= max_x + margin and 
            min_y - margin <= robot_y <= max_y + margin):
            relevant_map_indices.append(i)
    
    # For each LIDAR reading
    for azimuth, range_m in lidar_scan:
        if range_m >= lidar_config.max_range:
            continue
            
        # Calculate global position of the detected obstacle
        global_angle = azimuth + robot_theta
        obstacle_x = robot_x + range_m * np.cos(global_angle)
        obstacle_y = robot_y + range_m * np.sin(global_angle)
        
        # Update each relevant map
        for map_idx in relevant_map_indices:
            # Convert obstacle position to grid coordinates for this map
            grid_x, grid_y = maps.meters_to_grid(obstacle_x, obstacle_y)
            
            # Get map boundaries
            points_per_side = int(maps.config.map_size / maps.config.granularity)
            
            # Check if the obstacle falls within this map
            if (0 <= grid_x < points_per_side and 
                0 <= grid_y < points_per_side):
                
                # Mark the obstacle and surrounding cells (to account for obstacle size)
                obstacle_radius_cells = max(1, int(0.2 / maps.config.granularity))  # 20cm radius
                y_indices, x_indices = np.ogrid[-obstacle_radius_cells:obstacle_radius_cells + 1,
                                              -obstacle_radius_cells:obstacle_radius_cells + 1]
                circle_mask = x_indices**2 + y_indices**2 <= obstacle_radius_cells**2
                
                for dy in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                    for dx in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                        if not circle_mask[dy + obstacle_radius_cells, dx + obstacle_radius_cells]:
                            continue
                            
                        new_x = grid_x + dx
                        new_y = grid_y + dy
                        
                        if (0 <= new_x < points_per_side and 
                            0 <= new_y < points_per_side):
                            # Only update if cell isn't a permanent obstacle
                            if updated_maps.occupancy_maps[map_idx, new_y, new_x] != 1:
                                updated_maps.occupancy_maps[map_idx, new_y, new_x] = temp_obstacle_value
                
                # Also mark cells between robot and obstacle as free space
                # (ray tracing to clear space)
                robot_grid_x, robot_grid_y = maps.meters_to_grid(robot_x, robot_y)
                if (0 <= robot_grid_x < points_per_side and 
                    0 <= robot_grid_y < points_per_side):
                    # Use Bresenham's line algorithm
                    cells = bresenham_line(robot_grid_x, robot_grid_y, grid_x, grid_y)
                    for cell_x, cell_y in cells[:-1]:  # Exclude the last point (obstacle)
                        if (0 <= cell_x < points_per_side and 
                            0 <= cell_y < points_per_side):
                            # Only update if cell isn't a permanent obstacle
                            if updated_maps.occupancy_maps[map_idx, cell_y, cell_x] != 1:
                                updated_maps.occupancy_maps[map_idx, cell_y, cell_x] = 0
    
    return updated_maps

def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Implementation of Bresenham's line algorithm for ray tracing.
    Returns list of cells that the line passes through.
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    cells.append((x, y))
    return cells

# Example usage
if __name__ == "__main__":
    # Create sample map configuration
    map_config = MapConfig(
        num_waypoints=3,
        waypoint_distance=2.0,
        map_size=10.0,
        granularity=0.1
    )
    
    # Initialize maps
    maps = initialize_waypoint_maps(map_config)
    
    # Create LIDAR configuration
    lidar_config = LidarConfig(
        angular_resolution=1.0,
        max_range=10.0,
        min_range=0.1,
        noise_std=0.01
    )
    
    # Create sample obstacles
    obstacles = [
        Obstacle(x=2.0, y=0.0, radius=0.5)
    ]
    
    # Get a LIDAR scan
    robot_pose = (0.0, 0.0, 0.0)
    scan = simulate_lidar_scan(robot_pose, obstacles, lidar_config)
    
    # Update maps with LIDAR data
    updated_maps = update_maps_with_lidar(maps, scan, robot_pose, lidar_config)
    
    # Print some stats about the updates
    print("\nMap update statistics:")
    for i in range(map_config.num_waypoints):
        temp_obstacles = np.sum(updated_maps.occupancy_maps[i] == 2)
        free_space = np.sum(updated_maps.occupancy_maps[i] == 0)
        print(f"Map {i}: {temp_obstacles} temporary obstacles, {free_space} free cells marked") 
