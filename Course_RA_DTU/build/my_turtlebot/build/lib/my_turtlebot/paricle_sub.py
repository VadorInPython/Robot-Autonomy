#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_msgs.msg import ParticleCloud
from geometry_msgs.msg import PoseStamped
import math

class ParticleTracker(Node):
    def __init__(self):
        # CUSTOMIZE: You can change the node name here if desired
        super().__init__('particle_tracker_node')
        
        # CUSTOMIZE: Change the subscription topic if your particle cloud is published on a different topic
        self.subscription = self.create_subscription(
            ParticleCloud,
            '/particle_cloud',  # Default topic for AMCL particle cloud
            self.particle_callback,
            10)  # QoS profile depth
            
        # CUSTOMIZE: Change this topic name if you want to publish the expected pose to a different topic
        self.publisher = self.create_publisher(
            PoseStamped,
            '/expected_pose',  # Topic to publish the calculated expected pose
            10)  # QoS profile depth
            
        self.get_logger().info('Particle Tracker Node initialized')
        
    def particle_callback(self, msg):
        # Print the number of particles
        num_particles = len(msg.particles)
        self.get_logger().info(f'Number of particles: {num_particles}')
        
        if num_particles == 0:
            self.get_logger().warn('Received empty particle cloud')
            return
            
        # Compute the expected position using weighted average
        x_sum, y_sum, z_sum = 0.0, 0.0, 0.0
        qx_sum, qy_sum, qz_sum, qw_sum = 0.0, 0.0, 0.0, 0.0
        total_weight = 0.0
        
        for particle in msg.particles:
            weight = particle.weight
            
            # Position weighted sum
            x_sum += particle.pose.position.x * weight
            y_sum += particle.pose.position.y * weight
            z_sum += particle.pose.position.z * weight
            
            # Orientation weighted sum (quaternion components)
            qx_sum += particle.pose.orientation.x * weight
            qy_sum += particle.pose.orientation.y * weight
            qz_sum += particle.pose.orientation.z * weight
            qw_sum += particle.pose.orientation.w * weight
            
            total_weight += weight
        
        if total_weight > 0:
            # Calculate weighted average position
            expected_x = x_sum / total_weight
            expected_y = y_sum / total_weight
            expected_z = z_sum / total_weight
            
            # Calculate weighted average orientation
            expected_qx = qx_sum / total_weight
            expected_qy = qy_sum / total_weight
            expected_qz = qz_sum / total_weight
            expected_qw = qw_sum / total_weight
            
            # Normalize quaternion
            norm = math.sqrt(expected_qx**2 + expected_qy**2 + expected_qz**2 + expected_qw**2)
            if norm > 0:
                expected_qx /= norm
                expected_qy /= norm
                expected_qz /= norm
                expected_qw /= norm
            
            self.get_logger().info(f'Expected Position: x={expected_x:.2f}, y={expected_y:.2f}, z={expected_z:.2f}')
            
            # CUSTOMIZE: You can add additional metrics or calculations here
            
            # Publish the expected pose
            self.publish_expected_pose(msg.header, expected_x, expected_y, expected_z,
                                     expected_qx, expected_qy, expected_qz, expected_qw)
        else:
            self.get_logger().warn('Total particle weight is zero')
    
    def publish_expected_pose(self, header, x, y, z, qx, qy, qz, qw):
        pose_msg = PoseStamped()
        pose_msg.header = header
        
        # Set position
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        
        # Set orientation
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        
        # Publish the message
        self.publisher.publish(pose_msg)
        self.get_logger().debug('Published expected pose')

def main(args=None):
    rclpy.init(args=args)
    
    particle_tracker = ParticleTracker()
    
    try:
        # CUSTOMIZE: You can add additional functionality here before spinning the node
        rclpy.spin(particle_tracker)
    except KeyboardInterrupt:
        particle_tracker.get_logger().info('Node stopped cleanly')
    except Exception as e:
        particle_tracker.get_logger().error(f'Error: {str(e)}')
    finally:
        # Destroy the node explicitly
        particle_tracker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
