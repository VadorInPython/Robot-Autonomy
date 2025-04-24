import random
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import rclpy
from rclpy.node import Node

class PRMSampler(Node):
    def __init__(self):
        super().__init__('prm_sampler')
        
        # Subscribe to map
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        
        # Publisher for PRM markers
        self.marker_publisher = self.create_publisher(
            MarkerArray, 
            '/prm_markers', 
            10)
        
        self.map_data = None
        self.map_info = None
        
    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        # Generate samples when map is received
        self.generate_samples()
        
    def generate_samples(self, num_samples=8):
        if self.map_data is None:
            self.get_logger().info('No map data available yet')
            return
            
        samples = []
        attempts = 0
        max_attempts = 1000  # Prevent infinite loops
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate random indices within map bounds
            x_idx = random.randint(0, self.map_info.width - 1)
            y_idx = random.randint(0, self.map_info.height - 1)
            
            # Check if this is a valid location (not unknown=-1 or occupied=100)
            map_value = self.map_data[y_idx, x_idx]
            if map_value != -1 and map_value < 50:  # Free space threshold (typically < 50)
                # Convert to world coordinates
                x_world = x_idx * self.map_info.resolution + self.map_info.origin.position.x
                y_world = y_idx * self.map_info.resolution + self.map_info.origin.position.y
                
                samples.append((x_world, y_world))
                
        self.get_logger().info(f'Generated {len(samples)} valid samples out of {attempts} attempts')
        
        # Publish markers
        self.publish_markers(samples)
        
    def publish_markers(self, samples):
        marker_array = MarkerArray()
        
        # Create a marker for each sample
        for i, (x, y) in enumerate(samples):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "prm_nodes"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Set the position
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            
            # Set scale (size of sphere)
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            # Set color (blue)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            
        # Publish the marker array
        self.marker_publisher.publish(marker_array)
        self.get_logger().info('Published PRM node markers')

def main(args=None):
    rclpy.init(args=args)
    node = PRMSampler()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
