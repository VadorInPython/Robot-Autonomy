import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import numpy as np
import math
import os
import time
from transforms3d.euler import quat2euler
from threading import Lock

class IntegratedLocalizationMapper(Node):
    def __init__(self):
        super().__init__('integrated_localization_mapper')
        
        # Create QoS profiles
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Map parameters
        self.map_resolution = 0.05  # meters per cell
        self.map_width = 1000      # cells
        self.map_height = 1000     # cells
        self.map_origin_x = -25.0  # meters
        self.map_origin_y = -25.0  # meters
        
        # Counter-based grid parameters
        self.free_threshold = -5    # Counter value below which a cell is considered free
        self.occupied_threshold = 5 # Counter value above which a cell is considered occupied
        self.free_increment = -1    # Value to decrement counter when cell is observed as free
        self.occupied_increment = 2 # Value to increment counter when cell is observed as occupied
        self.counter_min = -10      # Minimum counter value
        self.counter_max = 10       # Maximum counter value
        
        # Initialize the counter grid with mutex for thread safety
        self.counter_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        self.occupancy_grid = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        self.grid_lock = Lock()
        
        # Track robot position with uncertainty
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.position_covariance = np.eye(3)  # x, y, yaw covariance
        self.initial_pose_received = False
        self.pose_lock = Lock()
        
        # Trajectory history for visualization
        self.trajectory = []
        self.max_trajectory_points = 1000  # Limit trajectory history
        
        # Create subscriptions
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            sensor_qos
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )
        
        # Subscribe to pose from AMCL if available (optional)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )
        
        # Publishers
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/integrated_map',
            10
        )
        
        self.trajectory_pub = self.create_publisher(
            MarkerArray,
            '/robot_trajectory',
            10
        )
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/estimated_pose',
            10
        )
        
        # Timers
        self.map_timer = self.create_timer(1.0, self.publish_map)
        self.trajectory_timer = self.create_timer(0.5, self.publish_trajectory)
        self.save_map_timer = self.create_timer(30.0, self.save_map)  # Save map every 30 seconds
        
        # Path for saving maps
        self.map_save_path = os.path.expanduser('~') + '/ros2_ws2/maps/'
        os.makedirs(self.map_save_path, exist_ok=True)
        
        self.get_logger().info('Integrated Localization and Mapping system initialized')
    
    def odom_callback(self, msg):
        # Extract position from odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to Euler angles
        _, _, yaw = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
        
        with self.pose_lock:
            # If this is the first message, initialize the robot's position
            if not self.initial_pose_received:
                self.initial_pose_received = True
                self.get_logger().info(f'Initial pose received: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}')
                
            # Update current position
            self.current_x = position.x
            self.current_y = position.y
            self.current_yaw = yaw
            
            # Extract position covariance
            pose_cov = np.array(msg.pose.covariance).reshape(6, 6)
            self.position_covariance[0, 0] = pose_cov[0, 0]  # x variance
            self.position_covariance[1, 1] = pose_cov[1, 1]  # y variance
            self.position_covariance[2, 2] = pose_cov[5, 5]  # yaw variance
            
            # Add point to trajectory
            self.trajectory.append((self.current_x, self.current_y, self.current_yaw))
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)
                
        # Publish the current estimated pose
        self.publish_pose()
    
    def pose_callback(self, msg):
        # This is triggered when we receive pose updates from a localization system like AMCL
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to Euler angles
        _, _, yaw = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
        
        with self.pose_lock:
            # Update current position with the localization estimate
            self.current_x = position.x
            self.current_y = position.y
            self.current_yaw = yaw
            
            # Extract position covariance
            pose_cov = np.array(msg.pose.covariance).reshape(6, 6)
            self.position_covariance[0, 0] = pose_cov[0, 0]  # x variance
            self.position_covariance[1, 1] = pose_cov[1, 1]  # y variance
            self.position_covariance[2, 2] = pose_cov[5, 5]  # yaw variance
            
            # Add point to trajectory
            self.trajectory.append((self.current_x, self.current_y, self.current_yaw))
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)
        
        # Log the updated position
        self.get_logger().debug(f'Localization update: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}')
        
        # Publish the current estimated pose
        self.publish_pose()
    
    def scan_callback(self, msg):
        if not self.initial_pose_received:
            return
        
        # Make a local copy of the current pose to ensure thread safety
        with self.pose_lock:
            current_x = self.current_x
            current_y = self.current_y
            current_yaw = self.current_yaw
            
        # Get laser parameters
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max
        ranges = msg.ranges
        
        # Process each laser beam
        for i, r in enumerate(ranges):
            # Calculate beam angle in robot frame
            angle = angle_min + i * angle_increment
            
            # Skip invalid measurements
            if not (range_min <= r <= range_max) or math.isnan(r) or math.isinf(r):
                continue
                
            # Calculate beam endpoint in world frame
            # First calculate in robot frame
            beam_x = r * math.cos(angle)
            beam_y = r * math.sin(angle)
            
            # Then transform to world frame
            world_beam_x = current_x + beam_x * math.cos(current_yaw) - beam_y * math.sin(current_yaw)
            world_beam_y = current_y + beam_x * math.sin(current_yaw) + beam_y * math.cos(current_yaw)
            
            # Mark the beam endpoint as occupied
            self.update_cell(world_beam_x, world_beam_y, is_occupied=True)
            
            # Mark cells along the beam as free using Bresenham's line algorithm
            self.mark_free_space(current_x, current_y, world_beam_x, world_beam_y)
    
    def update_cell(self, x, y, is_occupied):
        # Convert world coordinates to grid coordinates
        grid_x = int((x - self.map_origin_x) / self.map_resolution)
        grid_y = int((y - self.map_origin_y) / self.map_resolution)
        
        # Ensure coordinates are within map bounds
        if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
            with self.grid_lock:
                if is_occupied:
                    # Increment counter for occupied cells
                    self.counter_grid[grid_y, grid_x] = min(
                        self.counter_grid[grid_y, grid_x] + self.occupied_increment, 
                        self.counter_max
                    )
                else:
                    # Decrement counter for free cells
                    self.counter_grid[grid_y, grid_x] = max(
                        self.counter_grid[grid_y, grid_x] + self.free_increment, 
                        self.counter_min
                    )
                
                # Update occupancy grid based on counter value
                if self.counter_grid[grid_y, grid_x] <= self.free_threshold:
                    self.occupancy_grid[grid_y, grid_x] = 0  # Free
                elif self.counter_grid[grid_y, grid_x] >= self.occupied_threshold:
                    self.occupancy_grid[grid_y, grid_x] = 100  # Occupied
                else:
                    self.occupancy_grid[grid_y, grid_x] = 50  # Unknown/undecided
    
    def mark_free_space(self, x0, y0, x1, y1):
        # Convert world coordinates to grid coordinates
        grid_x0 = int((x0 - self.map_origin_x) / self.map_resolution)
        grid_y0 = int((y0 - self.map_origin_y) / self.map_resolution)
        grid_x1 = int((x1 - self.map_origin_x) / self.map_resolution)
        grid_y1 = int((y1 - self.map_origin_y) / self.map_resolution)
        
        # Bresenham's line algorithm to mark cells along the beam as free
        dx = abs(grid_x1 - grid_x0)
        dy = -abs(grid_y1 - grid_y0)
        sx = 1 if grid_x0 < grid_x1 else -1
        sy = 1 if grid_y0 < grid_y1 else -1
        err = dx + dy
        
        while True:
            # Don't mark the endpoint as free (which might be an obstacle)
            if grid_x0 == grid_x1 and grid_y0 == grid_y1:
                break
                
            # Mark cell as free
            if 0 <= grid_x0 < self.map_width and 0 <= grid_y0 < self.map_height:
                self.update_cell(
                    grid_x0 * self.map_resolution + self.map_origin_x,
                    grid_y0 * self.map_resolution + self.map_origin_y,
                    is_occupied=False
                )
                
            # Check if we've reached the endpoint
            if grid_x0 == grid_x1 and grid_y0 == grid_y1:
                break
                
            # Update position
            e2 = 2 * err
            if e2 >= dy:
                if grid_x0 == grid_x1:
                    break
                err += dy
                grid_x0 += sx
            if e2 <= dx:
                if grid_y0 == grid_y1:
                    break
                err += dx
                grid_y0 += sy
    
    def publish_map(self):
        if not self.initial_pose_received:
            return
            
        # Create OccupancyGrid message
        map_msg = OccupancyGrid()
        
        # Set header
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'
        
        # Set map metadata
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height
        
        # Set origin
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.x = 0.0
        map_msg.info.origin.orientation.y = 0.0
        map_msg.info.origin.orientation.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        # Copy the occupancy grid data with thread safety
        with self.grid_lock:
            # Flatten the 2D array to 1D array for the message
            map_msg.data = self.occupancy_grid.flatten().tolist()
        
        # Publish the map
        self.map_pub.publish(map_msg)
        self.get_logger().info('Integrated map published')
    
    def publish_pose(self):
        if not self.initial_pose_received:
            return
            
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        with self.pose_lock:
            pose_msg.pose.position.x = self.current_x
            pose_msg.pose.position.y = self.current_y
            pose_msg.pose.position.z = 0.0
            
            # Convert yaw back to quaternion
            pose_msg.pose.orientation.w = math.cos(self.current_yaw / 2.0)
            pose_msg.pose.orientation.z = math.sin(self.current_yaw / 2.0)
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
        
        # Publish the pose
        self.pose_pub.publish(pose_msg)
    
    def publish_trajectory(self):
        if not self.initial_pose_received or len(self.trajectory) < 2:
            return
            
        # Create marker array for trajectory visualization
        marker_array = MarkerArray()
        
        # Line strip for trajectory path
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'trajectory'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # Line width
        line_marker.color.r = 1.0
        line_marker.color.g = 0.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        line_marker.pose.orientation.w = 1.0
        
        # Copy trajectory with thread safety
        with self.pose_lock:
            # Add points to line_marker
            for x, y, _ in self.trajectory:
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.1  # Slightly above ground
                line_marker.points.append(p)
                
                # Add color (constant for line strip)
                color = ColorRGBA()
                color.r = 1.0
                color.g = 0.0
                color.b = 0.0
                color.a = 1.0
                line_marker.colors.append(color)
            
            # Current position marker
            if self.trajectory:
                pose_marker = Marker()
                pose_marker.header.frame_id = 'map'
                pose_marker.header.stamp = self.get_clock().now().to_msg()
                pose_marker.ns = 'current_pose'
                pose_marker.id = 1
                pose_marker.type = Marker.ARROW
                pose_marker.action = Marker.ADD
                pose_marker.scale.x = 0.5  # Arrow length
                pose_marker.scale.y = 0.1  # Arrow width
                pose_marker.scale.z = 0.1  # Arrow height
                pose_marker.color.r = 0.0
                pose_marker.color.g = 1.0
                pose_marker.color.b = 0.0
                pose_marker.color.a = 1.0
                
                current_x, current_y, current_yaw = self.trajectory[-1]
                pose_marker.pose.position.x = current_x
                pose_marker.pose.position.y = current_y
                pose_marker.pose.position.z = 0.1
                
                # Convert yaw to quaternion
                pose_marker.pose.orientation.w = math.cos(current_yaw / 2.0)
                pose_marker.pose.orientation.z = math.sin(current_yaw / 2.0)
                pose_marker.pose.orientation.x = 0.0
                pose_marker.pose.orientation.y = 0.0
                
                marker_array.markers.append(pose_marker)
        
        marker_array.markers.append(line_marker)
        self.trajectory_pub.publish(marker_array)
    
    def save_map(self):
        if not self.initial_pose_received:
            return
            
        # Create timestamp for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.map_save_path}integrated_map_{timestamp}"
        
        # Save map metadata
        metadata = {
            'resolution': self.map_resolution,
            'width': self.map_width,
            'height': self.map_height,
            'origin_x': self.map_origin_x,
            'origin_y': self.map_origin_y
        }
        
        # Save map data as numpy array
        with self.grid_lock:
            np.savez(filename, 
                     occupancy_grid=self.occupancy_grid,
                     counter_grid=self.counter_grid,
                     metadata=metadata)
        
        self.get_logger().info(f'Map saved to {filename}.npz')
        
        # Additionally, export as a PGM file for ROS map_server compatibility
        with self.grid_lock:
            # Convert to PGM format (0-255 grayscale)
            pgm_data = np.copy(self.occupancy_grid)
            # Convert -1 (unknown) to 205 (gray), 0 (free) to 254 (white), 100 (occupied) to 0 (black)
            pgm_data[pgm_data == -1] = 205
            pgm_data[pgm_data == 0] = 254
            pgm_data[pgm_data == 100] = 0
            
            # Save as PGM
            with open(f"{filename}.pgm", 'wb') as f:
                f.write(f"P5\n{self.map_width} {self.map_height}\n255\n".encode())
                pgm_data.astype(np.uint8).tofile(f)
                
            # Save YAML metadata for map_server
            with open(f"{filename}.yaml", 'w') as f:
                f.write(f"image: {os.path.basename(filename)}.pgm\n")
                f.write(f"resolution: {self.map_resolution}\n")
                f.write(f"origin: [{self.map_origin_x}, {self.map_origin_y}, 0.0]\n")
                f.write("occupied_thresh: 0.65\n")
                f.write("free_thresh: 0.196\n")
                f.write("negate: 0\n")
        
        self.get_logger().info(f'Map also exported as PGM/YAML for map_server at {filename}.pgm')

def main(args=None):
    rclpy.init(args=args)
    integrated_mapper = IntegratedLocalizationMapper()
    rclpy.spin(integrated_mapper)
    integrated_mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
