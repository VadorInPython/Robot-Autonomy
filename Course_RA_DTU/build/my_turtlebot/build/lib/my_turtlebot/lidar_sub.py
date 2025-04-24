import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10  # QoS profile depth
        )
        self.subscription  # prevent unused variable warning

    def lidar_callback(self, msg):
        if msg.ranges:
            min_distance = min([r for r in msg.ranges if r > 0.0])  # Exclude zero values
            self.get_logger().info(f'Minimum distance to obstacle: {min_distance:.2f} meters')
        else:
            self.get_logger().warn('No LIDAR data received!')


def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarSubscriber()
    rclpy.spin(lidar_subscriber)
    lidar_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
