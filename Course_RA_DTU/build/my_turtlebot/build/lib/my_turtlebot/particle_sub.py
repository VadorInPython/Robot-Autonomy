import rclpy
from rclpy.node import Node
from nav2_msgs.msg import ParticleCloud
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class ParticleSubscriber(Node):
    def __init__(self):
        super().__init__('particle_subscriber')
        
        # Create QoS profile to match AMCL's publisher
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Create subscription
        self.subscription = self.create_subscription(
            ParticleCloud,
            '/particle_cloud',
            self.particle_callback,
            qos_profile
        )
        
    def particle_callback(self, msg):
        # Print the number of particles
        num_particles = len(msg.particles)
        self.get_logger().info(f'Number of particles: {num_particles}')
        
        # Skip computation if no particles
        if num_particles == 0:
            return
            
        # Compute weighted average position
        x_sum, y_sum, total_weight = 0.0, 0.0, 0.0
        
        for particle in msg.particles:
            weight = particle.weight
            x_sum += particle.pose.position.x * weight
            y_sum += particle.pose.position.y * weight
            total_weight += weight
        
        # Calculate expected position
        if total_weight > 0:
            expected_x = x_sum / total_weight
            expected_y = y_sum / total_weight
            self.get_logger().info(f'Expected Position: x={expected_x:.2f}, y={expected_y:.2f}')

def main(args=None):
    rclpy.init(args=args)
    particle_subscriber = ParticleSubscriber()
    rclpy.spin(particle_subscriber)
    particle_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
