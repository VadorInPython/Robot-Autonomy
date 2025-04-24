"""
Script to create a simple map publisher node.

- Create a 2D occupancy grid using a 2D matrix
- Fill random cells in the matrix with the value 100 and the rest with 0
- Create an OccupancyGrid message and fill in the information (along with your map)
- Publish the map on the topic /map with a frequency of 0.5Hz
- Remember to add a transform that the map can live in (either static or dynamic)
"""
import rclpy
import random
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

PUB_NAME = "map_publisher"
PUB_MSG = OccupancyGrid
PUB_TOPIC = "/map"
PUB_FREQ = 2


class MapPublisher(Node):
    def __init__(self):
        super().__init__(PUB_NAME)
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(PUB_MSG, PUB_TOPIC, self.qos_profile)
        self.timer = self.create_timer(1 / PUB_FREQ, self.timer_callback)

    def timer_callback(self):
        # Prepare map metadata
        map_meta = MapMetaData()
        map_meta.resolution = 0.5  # m / cell
        map_meta.width = 10
        map_meta.height = 10
        map_meta.origin.position.x = -0.5 * map_meta.width * map_meta.resolution
        map_meta.origin.position.y = -0.5 * map_meta.height * map_meta.resolution

        # Randomize an obstacle in a 2D grid
        grid = [0] * (map_meta.width * map_meta.height)
        x = random.randint(0, map_meta.width)
        y = random.randint(0, map_meta.height)
        width = random.randint(1, int(map_meta.width / 2))
        height = random.randint(1, int(map_meta.height / 2))
        for i in range(x, x + width):
            for j in range(y, y + height):
                if i < map_meta.width and j < map_meta.height:
                    grid[i + j * map_meta.width] = 100

        # Publish message
        msg = OccupancyGrid()
        msg.info = map_meta
        msg.data = grid
        msg.header.frame_id = "map"
        self.publisher.publish(msg)
        self.get_logger().info("Publishing map, obstacle at ({}, {})".format(x, y))


def main(args=None):
    rclpy.init(args=args)
    pub = MapPublisher()
    rclpy.spin(pub)

    pub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
