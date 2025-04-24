import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import random
import numpy as np
import math

class PRMGenerator(Node):
    def __init__(self):
        super().__init__('prm_generator')
        
        # Parameters
        self.num_nodes = 10
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.prm_nodes = []
        self.node_info_gain = []
        self.current_position = (0.0, 0.0)
        self.ray_length = 5.0
        self.ray_step = 0.1
        self.angular_resolution = 10
        self.goal_sent = False
        self.current_goal_index = None
        self.info_gain_threshold = 1.0  # Info gain below which exploration stops
        
        # Create QoS profile for map subscription
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        
        self.marker_publisher = self.create_publisher(MarkerArray, '/prm_markers', 10)
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        
        self.timer = self.create_timer(2.0, self.main_loop)
        
        self.get_logger().info('PRM Generator initialized')

    def map_callback(self, msg):
        self.get_logger().info('Map received')
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

    def grid_to_world(self, grid_x, grid_y):
        return grid_x * self.map_resolution + self.map_origin_x, grid_y * self.map_resolution + self.map_origin_y

    def world_to_grid(self, world_x, world_y):
        return int((world_x - self.map_origin_x) / self.map_resolution), int((world_y - self.map_origin_y) / self.map_resolution)

    def is_valid_node(self, grid_x, grid_y):
        if grid_x < 0 or grid_x >= self.map_width or grid_y < 0 or grid_y >= self.map_height:
            return False
        return self.map_data[grid_y, grid_x] == 0

    def cast_ray(self, start_x, start_y, angle, max_distance):
        start_grid_x, start_grid_y = self.world_to_grid(start_x, start_y)
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        visible_unknown_cells = set()
        distance = 0.0
        while distance < max_distance:
            current_x = start_x + dir_x * distance
            current_y = start_y + dir_y * distance
            grid_x, grid_y = self.world_to_grid(current_x, current_y)
            if grid_x < 0 or grid_x >= self.map_width or grid_y < 0 or grid_y >= self.map_height:
                break
            cell_value = self.map_data[grid_y, grid_x]
            if cell_value == 100:
                break
            if cell_value == -1:
                visible_unknown_cells.add((grid_x, grid_y))
            distance += self.ray_step
        return visible_unknown_cells

    def compute_visible_cells(self, node_x, node_y):
        visible_cells = set()
        for i in range(0, 360, self.angular_resolution):
            angle_rad = math.radians(i)
            visible_cells.update(self.cast_ray(node_x, node_y, angle_rad, self.ray_length))
        return visible_cells

    def compute_information_gain(self, node_index):
        node_x, node_y = self.prm_nodes[node_index]
        visible_cells = self.compute_visible_cells(node_x, node_y)
        visible_count = len(visible_cells)
        if node_index > 0:
            prev_node_x, prev_node_y = self.prm_nodes[node_index - 1]
        else:
            prev_node_x, prev_node_y = self.current_position
        distance = math.sqrt((node_x - prev_node_x)**2 + (node_y - prev_node_y)**2)
        lambda_param = 1.0
        info_gain = visible_count - lambda_param * distance
        return info_gain, visible_count, distance

    def generate_prm(self):
        self.prm_nodes = []
        attempts = 0
        max_attempts = 1000
        while len(self.prm_nodes) < self.num_nodes and attempts < max_attempts:
            grid_x = random.randint(0, self.map_width - 1)
            grid_y = random.randint(0, self.map_height - 1)
            if self.is_valid_node(grid_x, grid_y):
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                self.prm_nodes.append((world_x, world_y))
            attempts += 1
        self.node_info_gain = []
        for i in range(len(self.prm_nodes)):
            info_gain, visible_count, distance = self.compute_information_gain(i)
            self.node_info_gain.append((i, info_gain, visible_count, distance))
        self.node_info_gain.sort(key=lambda x: x[1], reverse=True)
        self.publish_markers()

    def send_navigation_goal(self, x, y):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        self.nav_to_pose_client.wait_for_server()
        self.nav_to_pose_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)
        self.get_logger().info(f"Sent navigation goal to ({x:.2f}, {y:.2f})")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return
        self.get_logger().info("Goal accepted")
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info("Goal reached! Resampling PRM...")
        self.current_position = self.prm_nodes[self.current_goal_index]
        self.goal_sent = False
        self.current_goal_index = None

    def navigate_to_best_node(self):
        if not self.node_info_gain:
            return
        if self.node_info_gain[0][1] < self.info_gain_threshold:
            self.get_logger().info("All nodes have low information gain. Stopping exploration.")
            rclpy.shutdown()
            return
        best_node_index = self.node_info_gain[0][0]
        self.current_goal_index = best_node_index
        best_node = self.prm_nodes[best_node_index]
        self.send_navigation_goal(best_node[0], best_node[1])
        self.goal_sent = True

    def publish_markers(self):
        marker_array = MarkerArray()
        node_marker = Marker()
        node_marker.header.frame_id = "map"
        node_marker.header.stamp = self.get_clock().now().to_msg()
        node_marker.ns = "prm_nodes"
        node_marker.id = 0
        node_marker.type = Marker.SPHERE_LIST
        node_marker.action = Marker.ADD
        node_marker.pose.orientation.w = 1.0
        node_marker.scale.x = 0.5
        node_marker.scale.y = 0.5
        node_marker.scale.z = 0.5
        node_marker.color.r = 0.0
        node_marker.color.g = 0.0
        node_marker.color.b = 0.0
        node_marker.color.a = 1.0
        for node in self.prm_nodes:
            pt = Point()
            pt.x, pt.y, pt.z = node[0], node[1], 0.1
            node_marker.points.append(pt)
        marker_array.markers.append(node_marker)

        for rank, (i, info_gain, _, _) in enumerate(self.node_info_gain):
            node_x, node_y = self.prm_nodes[i]
            id_marker = Marker()
            id_marker.header.frame_id = "map"
            id_marker.header.stamp = self.get_clock().now().to_msg()
            id_marker.ns = "prm_node_ids"
            id_marker.id = i
            id_marker.type = Marker.TEXT_VIEW_FACING
            id_marker.action = Marker.ADD
            id_marker.pose.position.x = node_x
            id_marker.pose.position.y = node_y
            id_marker.pose.position.z = 0.3
            id_marker.pose.orientation.w = 1.0
            id_marker.scale.z = 0.3
            id_marker.color.r = 1.0
            id_marker.color.g = 1.0
            id_marker.color.b = 1.0
            id_marker.color.a = 1.0
            id_marker.text = f"{i} (Rank {rank+1})"
            marker_array.markers.append(id_marker)

            info_marker = Marker()
            info_marker.header.frame_id = "map"
            info_marker.header.stamp = self.get_clock().now().to_msg()
            info_marker.ns = "prm_node_info"
            info_marker.id = i + 100
            info_marker.type = Marker.TEXT_VIEW_FACING
            info_marker.action = Marker.ADD
            info_marker.pose.position.x = node_x
            info_marker.pose.position.y = node_y
            info_marker.pose.position.z = 0.6
            info_marker.pose.orientation.w = 1.0
            info_marker.scale.z = 0.2
            norm_rank = rank / max(1, len(self.node_info_gain) - 1)
            info_marker.color.r = norm_rank
            info_marker.color.g = 1.0 - norm_rank
            info_marker.color.b = 0.0
            info_marker.color.a = 1.0
            info_marker.text = f"IG: {info_gain:.2f}"
            marker_array.markers.append(info_marker)

        self.marker_publisher.publish(marker_array)

    def main_loop(self):
        if self.map_data is None:
            self.get_logger().info("Waiting for map...")
            return

        if not self.goal_sent:
            self.generate_prm()
            self.navigate_to_best_node()

def main(args=None):
    rclpy.init(args=args)
    node = PRMGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

