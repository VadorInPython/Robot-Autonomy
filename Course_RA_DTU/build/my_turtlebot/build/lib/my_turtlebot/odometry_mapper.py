import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import numpy as np
import math
from transforms3d.euler import quat2euler

class OdometryMapper(Node):
    def __init__(self):
        super().__init__('odometry_mapper')
        
        # Create QoS profile for odometry subscription
        odom_qos = QoSProfile(
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
        
        # Initialize the map (0 = unknown, 100 = occupied, -1 = free)
        self.map_data = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        
        # Track robot position
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.initial_pose_received = False
        
        # Create odometry subscription
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            odom_qos
        )
        
        # Create map publisher
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/odometry_map',
            10
        )
        
        # Timer for publishing the map
        self.map_timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info('Odometry Mapper initialized')
    
    def odom_callback(self, msg):
        # Extract position from odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to Euler angles using transforms3d
        # Note: quat2euler expects [w, x, y, z] order while ROS uses [x, y, z, w]
        _, _, yaw = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
        
        # If this is the first message, initialize the robot's position
        if not self.initial_pose_received:
            self.initial_pose_received = True
            self.get_logger().info(f'Initial pose received: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}')
            
        # Update current position
        self.current_x = position.x
        self.current_y = position.y
        self.current_yaw = yaw
        
        # Update map with current position (mark as occupied)
        self.update_map(self.current_x, self.current_y)
        
    def update_map(self, x, y):
        # Convert world coordinates to grid coordinates
        grid_x = int((x - self.map_origin_x) / self.map_resolution)
        grid_y = int((y - self.map_origin_y) / self.map_resolution)
        
        # Ensure coordinates are within map bounds
        if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
            # Mark the cell as occupied (100)
            self.map_data[grid_y, grid_x] = 100
            
            # Mark a small area around the robot as free space (-1)
            radius = 3  # cells
            for i in range(max(0, grid_y - radius), min(self.map_height, grid_y + radius + 1)):
                for j in range(max(0, grid_x - radius), min(self.map_width, grid_x + radius + 1)):
                    if (i != grid_y or j != grid_x) and self.map_data[i, j] != 100:
                        self.map_data[i, j] = 0  # Free space
    
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
        map_msg.data = self.map_data.flatten().tolist()
        
        # Publish the map
        self.map_pub.publish(map_msg)
        self.get_logger().info(f'Map published. Robot at: x={self.current_x:.2f}, y={self.current_y:.2f}')

def main(args=None):
    rclpy.init(args=args)
    odometry_mapper = OdometryMapper()
    rclpy.spin(odometry_mapper)
    odometry_mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
