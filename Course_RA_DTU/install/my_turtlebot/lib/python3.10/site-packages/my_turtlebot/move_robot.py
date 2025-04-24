"""
Simple node to move the turtlebot constant +x velocity.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions


class MoveTurtlebot(Node):
    def __init__(self):
        super().__init__("simple_publisher")
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(Twist, "/cmd_vel", self.qos_profile)
        freq = 5
        self.timer = self.create_timer(1 / freq, self.timer_callback)

    def timer_callback(self):
        # Publish the message
        msg = Twist()
        msg.linear.x = 0.05
        self.publisher.publish(msg)
        self.get_logger().info(f"Publishing x-vel: {msg.linear.x}", once=True)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    pub = MoveTurtlebot()
    try:
        rclpy.spin(pub)
    except KeyboardInterrupt:
        msg = Twist()
        msg.linear.x = 0.0
        pub.publisher.publish(msg)
        pub.get_logger().info(f"Publishing x-vel: {msg.linear.x}")
        pub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
