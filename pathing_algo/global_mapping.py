import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

@dataclass
class MapConfig:
    """Configuration class for the tour robot map system."""
    num_waypoints: int
    waypoint_distance: float  # meters
    map_size: float  # meters
    granularity: float  # meters per point

@dataclass
class WaypointMaps:
    """Class containing both the occupancy maps and waypoint locations."""
    occupancy_maps: np.ndarray  # Shape: (num_waypoints, points_per_side, points_per_side)
    waypoint_indices: np.ndarray  # Shape: (num_waypoints, 2, 2) for [[prev_xy, curr_xy], ...]
    config: MapConfig  # Store config for unit conversion calculations
    
    def grid_to_meters(self, grid_x: Union[int, np.ndarray], grid_y: Union[int, np.ndarray]) -> Tuple[float, float]:
        """
        Convert grid coordinates to meters from the map origin.
        
        Parameters
        ----------
        grid_x : int or numpy.ndarray
            X coordinate(s) in grid cells
        grid_y : int or numpy.ndarray
            Y coordinate(s) in grid cells
            
        Returns
        -------
        Tuple[float, float]
            (x_meters, y_meters) from origin of the map
        """
        # Origin is at (0,0) of the grid
        x_meters = grid_x * self.config.granularity
        y_meters = grid_y * self.config.granularity
        return x_meters, y_meters
    
    def meters_to_grid(self, x_meters: float, y_meters: float) -> Tuple[int, int]:
        """
        Convert coordinates in meters to closest grid cell indices.
        
        Parameters
        ----------
        x_meters : float
            X coordinate in meters
        y_meters : float
            Y coordinate in meters
            
        Returns
        -------
        Tuple[int, int]
            (grid_x, grid_y) indices of the closest grid cell
        """
        grid_x = int(round(x_meters / self.config.granularity))
        grid_y = int(round(y_meters / self.config.granularity))
        points_per_side = int(self.config.map_size / self.config.granularity)
        
        # Ensure we stay within grid bounds
        grid_x = np.clip(grid_x, 0, points_per_side - 1)
        grid_y = np.clip(grid_y, 0, points_per_side - 1)
        return grid_x, grid_y
    
    def get_waypoint_meters(self, waypoint_idx: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get the previous and current waypoint positions in meters for a given map.
        
        Parameters
        ----------
        waypoint_idx : int
            Index of the waypoint map
            
        Returns
        -------
        Tuple[Tuple[float, float], Tuple[float, float]]
            ((prev_x_meters, prev_y_meters), (curr_x_meters, curr_y_meters))
        """
        prev_grid = self.waypoint_indices[waypoint_idx, 0]
        curr_grid = self.waypoint_indices[waypoint_idx, 1]
        
        prev_meters = self.grid_to_meters(prev_grid[0], prev_grid[1])
        curr_meters = self.grid_to_meters(curr_grid[0], curr_grid[1])
        
        return prev_meters, curr_meters
    
    def get_map_boundaries_meters(self, waypoint_idx: int) -> Tuple[float, float, float, float]:
        """
        Get the boundaries of a specific map in meters.

        Parameters
        ----------
        waypoint_idx : int
            Index of the waypoint map

        Returns
        -------
        Tuple[float, float, float, float]
            (min_x_meters, max_x_meters, min_y_meters, max_y_meters)
        """
        points_per_side = int(self.config.map_size / self.config.granularity)
        min_x, min_y = self.grid_to_meters(0, 0)
        max_x, max_y = self.grid_to_meters(points_per_side - 1, points_per_side - 1)
        return min_x, max_x, min_y, max_y

def initialize_waypoint_maps(config: MapConfig) -> WaypointMaps:
    """
    Initialize both the 3D occupancy maps and the waypoint location indices.
    
    Parameters
    ----------
    config : MapConfig
        Configuration object containing map parameters
    
    Returns
    -------
    WaypointMaps
        A dataclass containing the occupancy maps and waypoint information
    """
    points_per_side = int(config.map_size / config.granularity)
    
    # Initialize the occupancy maps
    occupancy_maps = np.full(
        (config.num_waypoints, points_per_side, points_per_side),
        -1,
        dtype=np.int8
    )
    
    # Initialize the waypoint indices array
    waypoint_indices = np.zeros((config.num_waypoints, 2, 2), dtype=np.int32)
    
    # Calculate waypoint positions
    center_idx = points_per_side // 2
    
    for i in range(config.num_waypoints):
        waypoint_indices[i, 1] = [center_idx, center_idx]
        if i == 0:
            waypoint_indices[i, 0] = [center_idx, center_idx]
        else:
            waypoint_indices[i, 0] = waypoint_indices[i-1, 1]
    
    return WaypointMaps(occupancy_maps, waypoint_indices, config)

# Example usage
if __name__ == "__main__":
    config = MapConfig(
        num_waypoints=5,
        waypoint_distance=2.0,
        map_size=10.0,
        granularity=0.1
    )
    
    maps = initialize_waypoint_maps(config)
    
    # Example coordinate conversions
    grid_x, grid_y = 25, 25
    meters_x, meters_y = maps.grid_to_meters(grid_x, grid_y)
    print(f"Grid coordinates ({grid_x}, {grid_y}) = {meters_x:.2f}, {meters_y:.2f} meters")
    
    back_to_grid_x, back_to_grid_y = maps.meters_to_grid(meters_x, meters_y)
    print(f"Converting back: ({meters_x:.2f}, {meters_y:.2f}) meters = grid ({back_to_grid_x}, {back_to_grid_y})")
    
    # Get waypoint positions in meters
    prev_meters, curr_meters = maps.get_waypoint_meters(2)
    print(f"Waypoint 2: Previous at {prev_meters}m, Current at {curr_meters}m")
    
    # Get map boundaries
    min_x, max_x, min_y, max_y = maps.get_map_boundaries_meters(0)
    print(f"Map 0 boundaries: X: [{min_x:.2f}, {max_x:.2f}]m, Y: [{min_y:.2f}, {max_y:.2f}]m")
