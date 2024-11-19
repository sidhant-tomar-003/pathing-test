import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

@dataclass
class LidarConfig:
    """Configuration class for the LIDAR simulator."""
    angular_resolution: float  # degrees per reading (e.g., 1.0 means 360 readings)
    max_range: float  # meters
    min_range: float  # meters
    noise_std: float  # standard deviation for noise in meters
    
@dataclass
class Obstacle:
    """Class representing a simple obstacle for LIDAR simulation."""
    x: float  # meters
    y: float  # meters
    radius: float  # meters

def simulate_lidar_scan(
    robot_pose: Tuple[float, float, float],  # (x, y, theta) in meters and radians
    obstacles: List[Obstacle],
    config: LidarConfig
) -> np.ndarray:
    """
    Simulate a LIDAR scan from the robot's current pose, detecting given obstacles.
    
    Parameters
    ----------
    robot_pose : Tuple[float, float, float]
        Robot's current pose as (x, y, theta) where:
        - x, y are in meters
        - theta is in radians (0 is along positive x-axis)
    obstacles : List[Obstacle]
        List of obstacles in the environment
    config : LidarConfig
        LIDAR configuration parameters
        
    Returns
    -------
    numpy.ndarray
        Shape: (N, 2) where N = 360/angular_resolution
        Each row contains [azimuth (radians), range (meters)]
        - Azimuth: 0 to 2π, counterclockwise from robot's forward direction
        - Range: Distance to nearest obstacle or max_range if none detected
        
    Example
    -------
    >>> config = LidarConfig(
    ...     angular_resolution=1.0,  # 360 readings
    ...     max_range=10.0,
    ...     min_range=0.1,
    ...     noise_std=0.01
    ... )
    >>> robot_pose = (0, 0, 0)  # At origin, facing positive x
    >>> obstacles = [Obstacle(x=2.0, y=0.0, radius=0.5)]  # Single obstacle
    >>> scan = simulate_lidar_scan(robot_pose, obstacles, config)
    >>> print(f"Number of readings: {len(scan)}")  # Will print "Number of readings: 360"
    """
    # Calculate number of readings based on angular resolution
    num_readings = int(360 / config.angular_resolution)
    
    # Initialize output array [azimuth, range]
    scan_data = np.zeros((num_readings, 2))
    
    # Generate azimuth angles (convert to radians)
    scan_data[:, 0] = np.linspace(0, 2*np.pi, num_readings, endpoint=False)
    
    # Fill with max range initially
    scan_data[:, 1] = config.max_range
    
    robot_x, robot_y, robot_theta = robot_pose
    
    # For each azimuth angle
    for i, azimuth in enumerate(scan_data[:, 0]):
        # Global angle of the beam
        global_angle = azimuth + robot_theta
        
        # For each obstacle
        for obstacle in obstacles:
            # Vector from robot to obstacle center
            dx = obstacle.x - robot_x
            dy = obstacle.y - robot_y
            
            # Distance to obstacle center
            distance_to_center = np.sqrt(dx**2 + dy**2)
            
            # Angle to obstacle center
            angle_to_center = np.arctan2(dy, dx) - robot_theta
            angle_to_center = (angle_to_center + np.pi) % (2*np.pi) - np.pi  # Normalize to [-π, π]
            
            # Angular difference between beam and obstacle
            angle_diff = np.abs((azimuth - angle_to_center + np.pi) % (2*np.pi) - np.pi)
            
            # If beam might hit obstacle
            if angle_diff < np.pi/2:
                # Perpendicular distance from center to beam
                perp_distance = distance_to_center * np.sin(angle_diff)
                
                # If beam intersects obstacle
                if perp_distance <= obstacle.radius:
                    # Calculate intersection distance using pythagorean theorem
                    intersection_distance = np.sqrt(distance_to_center**2 - perp_distance**2) - \
                                         np.sqrt(obstacle.radius**2 - perp_distance**2)
                    
                    # Update range if this is closer than previous readings
                    if intersection_distance < scan_data[i, 1]:
                        scan_data[i, 1] = max(intersection_distance, config.min_range)
    
    # Add Gaussian noise to range measurements
    if config.noise_std > 0:
        scan_data[:, 1] += np.random.normal(0, config.noise_std, num_readings)
        # Clip to valid range
        scan_data[:, 1] = np.clip(scan_data[:, 1], config.min_range, config.max_range)
    
    return scan_data

# Example usage
if __name__ == "__main__":
    # Configure LIDAR
    config = LidarConfig(
        angular_resolution=1.0,  # 1 degree per reading (360 total readings)
        max_range=10.0,         # 10 meters max range
        min_range=0.1,          # 10 cm minimum range
        noise_std=0.01          # 1 cm noise standard deviation
    )
    
    # Create some sample obstacles
    obstacles = [
        Obstacle(x=2.0, y=0.0, radius=0.5),    # Obstacle directly in front
        Obstacle(x=0.0, y=3.0, radius=0.3),    # Obstacle to the left
        Obstacle(x=-2.0, y=-1.0, radius=0.4),  # Obstacle behind and right
    ]
    
    # Simulate a scan with robot at origin facing positive x-axis
    robot_pose = (0.0, 0.0, 0.0)
    scan = simulate_lidar_scan(robot_pose, obstacles, config)
    
    # Print some sample readings
    print("\nSample LIDAR readings (azimuth in degrees, range in meters):")
    angles = [0, 90, 180, 270]  # Check cardinal directions
    for angle in angles:
        idx = int(angle / config.angular_resolution)
        azimuth_deg = np.degrees(scan[idx, 0])
        range_m = scan[idx, 1]
        print(f"Angle: {azimuth_deg:>3.0f}°, Range: {range_m:.2f}m")
