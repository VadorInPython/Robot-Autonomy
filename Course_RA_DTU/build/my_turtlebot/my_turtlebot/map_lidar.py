import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry

SUB_NAME = "map_lidar"
SUB_MSG = LaserScan
SUB_TOPIC = "/scan"

PUB_MSG = OccupancyGrid
PUB_TOPIC = "/map2"
PUB_FREQ = 0.5

ODOM_NAME = "odom"
ODOM_MSG = Odometry
ODOM_TOPIC = "/odom"


class LidarSubscriber(Node):
    def __init__(self):
        super().__init__(SUB_NAME)
        scan_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        map_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self.scan_sub = self.create_subscription(
            SUB_MSG, SUB_TOPIC, self.scan_callback, scan_profile
        )

        self.odom_sub = self.create_subscription(
            ODOM_MSG, ODOM_TOPIC, self.odom_callback, scan_profile
        )

        self.publisher = self.create_publisher(
            PUB_MSG, PUB_TOPIC, map_profile
        )

        self.map_meta = self.get_map_metadata()
        # Create an empty 2D map
        self.grid = np.full((self.map_meta.height, self.map_meta.width),
                            -1, dtype=np.int8)

        # Odometry
        self.robot_x = 0.
        self.robot_y = 0.
        self.robot_heading = 0.

    def convert_scan_to_cloud(self, scan: LaserScan):
        """
        Convert the LIDAR scan data to euclidean np.arrays.

        Args:
            scan (LaserScan): The old LIDAR scan data.

        Returns:
            p (np.array): The point cloud data, nx2.
        """
        n_scans = len(scan.ranges)  # 360
        ranges = np.array(scan.ranges, np.float32)

        # Find indices where the scan data is inf
        inf_idx = np.isinf(ranges)
        ranges[inf_idx] = 0

        angle_min = scan.angle_min + self.robot_heading # IMPT FOR ACCUMULATION
        angle_increment = scan.angle_increment
        p = np.zeros((n_scans, 2))
        for i in range(n_scans):
            p[i, 0] = ranges[i] * np.cos(angle_min + angle_increment * i)
            p[i, 1] = ranges[i] * np.sin(angle_min + angle_increment * i)

        # Remove values from inf range
        p = np.delete(p, inf_idx, axis=0)
        return p

    def get_map_metadata(self) -> MapMetaData:
        """
        Prepare map metadata.
        """
        map_meta = MapMetaData()
        map_meta.resolution = 0.05  # m / cell

        map_height = 15     # in meters
        map_width = 15      # in meters
        map_meta.height = int(map_height / map_meta.resolution)
        map_meta.width = int(map_width / map_meta.resolution)
        # Orientation values were found after trial end error
        map_meta.origin.orientation.x = 0.7071068
        map_meta.origin.orientation.y = 0.7071068
        map_meta.origin.orientation.z = 0.0
        map_meta.origin.orientation.w = 0.0
        # Position values were found after trial and error
        map_meta.origin.position.x = -7.5
        map_meta.origin.position.y = -7.5
        map_meta.origin.position.z = 0.0
        return map_meta

    def draw_points(self, p, robot_gridx: int, robot_gridy: int) -> None:
        """
        Draw the points on the map.

        Args:
            p (np.array): The point cloud data with the robot at the center.
            map_meta (MapMetaData): The map metadata.
            robot_gridx (int): The robot's grid x position.
            robot_gridy (int): The robot's grid y position.
        """
        unit = 10       # increment/decrement whenever a cell is occupied/free
        for point in p:
            # Occupied
            # IMPT FOR ACCUMULATION
            grid_x = int(point[0] / self.map_meta.resolution) + robot_gridx # +robot's translation
            grid_y = int(point[1] / self.map_meta.resolution) + robot_gridy
            prev_value = self.grid[grid_x, grid_y]
            # self.grid[grid_x, grid_y] = 100  # occupied
            self.grid[grid_x, grid_y] += unit  # occupied
            self.grid[grid_x, grid_y] = min(100, prev_value+unit)  # cap at 100

            # Free
            free_area = bresenham((robot_gridx, robot_gridy), (grid_x, grid_y))
            for free in free_area[:-1]:  # last element contains obastacle
                prev_value = self.grid[grid_x, grid_y]
                self.grid[free[0], free[1]] = 0

    def odom_callback(self, msg: Odometry):
        """
        Update the robot's position.
        """
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Convert quarternion to euler
        x, y, z, w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, \
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        self.robot_heading = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))

    def scan_callback(self, scan: LaserScan) -> None:
        """
        Process the LIDAR sensor data.
        """

        # Convert the LIDAR scan data to euclidean array
        p = self.convert_scan_to_cloud(scan)  # where (0,0) is robot position

        # Assume robot is at the center of the grid map
        # IMPT FOR ACCUMULATION
        robot_gridx = self.map_meta.width / 2   # robot's initial position
        robot_gridy = self.map_meta.height / 2
        robot_gridx += self.robot_x / self.map_meta.resolution  # +translation from movement
        robot_gridy += self.robot_y / self.map_meta.resolution
        self.draw_points(p, int(robot_gridx), int(robot_gridy))

        # Count the number of occupied cells
        throttle_duration = 1  # in seconds
        n_occupied = np.sum(self.grid == 100)
        n_free = np.sum(self.grid == 0)
        n_unknown = np.sum(self.grid == -1)
        n_total = self.grid.shape[0] * self.grid.shape[1]
        log_str = f"{throttle_duration}s throttle \nOccupied: {n_occupied}, {n_occupied/n_total:.3f}%"
        log_str += f"\nFree: {n_free}, {n_free/n_total:.3f}%"
        log_str += f"\nUnknown: {n_unknown}, {n_unknown/n_total:.3f}%"
        self.get_logger().info(
            log_str,
            throttle_duration_sec=throttle_duration,
        )

        # Publish message
        msg = OccupancyGrid()
        msg.info = self.map_meta
        msg.data = self.grid.flatten().astype(np.int8).tolist()
        msg.header.frame_id = "map"
        self.publisher.publish(msg)


def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end including.

    (original from roguebasin.com)
    >> points1 = bresenham((4, 4), (6, 10))
    >> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


def main():
    rclpy.init()
    sub = LidarSubscriber()
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
