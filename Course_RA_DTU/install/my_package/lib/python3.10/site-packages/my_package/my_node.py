#! /usr/bin/env python3

"""
Spawn 10 turtles using the turtlesim Spawn service.

turtlesim/Spawn
float32 x
float32 y
float32 theta
string name # Optional.  A unique name will be created and returned if this is empty
---
string name

turtlesim/Kill
string name
---
"""
import rclpy
import random
from rclpy.node import Node
from turtlesim.srv import Spawn
from turtlesim.srv import Kill
from geometry_msgs.msg import Twist
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


class SpawnTurtle(Node):
    """A node to spawn 10 turtles and make them move in a circle motion."""

    def __init__(self):
        super().__init__("spawn_turtle")
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.get_logger().info("spawn_turtle node has been created")
        self.total = 10  # number of turtles to spawn

        # Create a client for the 'spawn' service
        self.spawn_cli = self.create_client(Spawn, "/spawn")
        while not self.spawn_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("spawn service not available, waiting again...")
        self.count = 0

        # Create a client for the 'kill' service
        self.kill_cli = self.create_client(Kill, "/kill")
        while not self.kill_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("kill service not available, waiting again...")

        # Create multiple publishers for cmd_vel messages
        self.cmd_publishers = []
        for i in range(self.total):
            topic = "T" + str(i + 1) + "/cmd_vel"
            self.cmd_publishers.append(
                self.create_publisher(Twist, topic, qos_profile=self.qos_profile)
            )
        freq = 5  # Hz

        # Create a timer to publish cmd_vel messages
        self.cmd_timer = self.create_timer(1 / freq, self.cmd_callback)

    def send_spawn_request(self, x: float, y: float, theta=0.0):
        """Send a request to spawn a turtle at the given position."""
        self.count += 1
        spawn_req = Spawn.Request()
        spawn_req.name = "T" + str(self.count)
        spawn_req.x = x
        spawn_req.y = y
        spawn_req.theta = theta
        self.future = self.spawn_cli.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def send_kill_request(self, name: str):
        """Send a request to kill a turtle with the given name."""
        kill_req = Kill.Request()
        kill_req.name = name
        self.future = self.kill_cli.call_async(kill_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def cmd_callback(self):
        """Make the turtles move in a circle motion."""
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.2
        for publisher in self.cmd_publishers:
            publisher.publish(msg)


def main(args=None):
    # Init node
    rclpy.init(args=args)
    spawn_turtle = SpawnTurtle()

    # Remove default turtle
    response = spawn_turtle.send_kill_request("turtle1")
    spawn_turtle.get_logger().info("Turtle successfully killed with turtle1")

    # Spawn 10 new turtles
    for i in range(spawn_turtle.total):
        x = random.randint(2, 8)
        y = random.randint(2, 8)
        response = spawn_turtle.send_spawn_request(float(x), float(y))
        spawn_turtle.get_logger().info(
            "Turtle successfully spawned with name %s" % response.name
        )

    # Spin node to publish cmd__vel messages continuosly
    rclpy.spin(spawn_turtle)

    # Destroy at the end
    spawn_turtle.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
