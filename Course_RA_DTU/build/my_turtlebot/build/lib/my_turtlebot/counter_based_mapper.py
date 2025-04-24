import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point, Quaternion
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import numpy as np
import math
from transforms3d.euler import quat2euler

class CounterBasedMapper(Node):
    def __init__(self):
        super().__init__('counter_based_mapper')
        
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
        
        # Initialize the counter grid
        self.counter_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        
        # Initialize the occupancy grid for publishing (0-100 scale)
        self.occupancy_grid = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        
        # Track robot position
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.initial_pose_received = False
        
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
        
        # Create map publisher
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/counter_map',
            10
        )
        
        # Timer for publishing the map
        self.map_timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info('Counter-Based Mapper initialized')
    
    def odom_callback(self, msg):
        # Extract position from odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to Euler angles using transforms3d
        _, _, yaw = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
        
        # If this is the first message, initialize the robot's position
        if not self.initial_pose_received:
            self.initial_pose_received = True
            self.get_logger().info(f'Initial pose received: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}')
            
        # Update current position
        self.current_x = position.x
        self.current_y = position.y
        self.current_yaw = yaw
    
    def scan_callback(self, msg):
        if not self.initial_pose_received:
            return
            
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
            world_beam_x = self.current_x + beam_x * math.cos(self.current_yaw) - beam_y * math.sin(self.current_yaw)
            world_beam_y = self.current_y + beam_x * math.sin(self.current_yaw) + beam_y * math.cos(self.current_yaw)
            
            # Mark the beam endpoint as occupied
            self.update_cell(world_beam_x, world_beam_y, is_occupied=True)
            
            # Mark cells along the beam as free using Bresenham's line algorithm
            self.mark_free_space(self.current_x, self.current_y, world_beam_x, world_beam_y)
    
    def update_cell(self, x, y, is_occupied):
        # Convert world coordinates to grid coordinates
        grid_x = int((x - self.map_origin_x) / self.map_resolution)
        grid_y = int((y - self.map_origin_y) / self.map_resolution)
        
        # Ensure coordinates are within map bounds
        if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
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
            # Don't mark the endpoint as free
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
        
        # Flatten the 2D array to 1D array for the message
        map_msg.data = self.occupancy_grid.flatten().tolist()
        
        # Publish the map
        self.map_pub.publish(map_msg)
        self.get_logger().info(f'Counter-based map published. Robot at: x={self.current_x:.2f}, y={self.current_y:.2f}')

def main(args=None):
    rclpy.init(args=args)
    counter_based_mapper = CounterBasedMapper()
    rclpy.spin(counter_based_mapper)
    counter_based_mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
