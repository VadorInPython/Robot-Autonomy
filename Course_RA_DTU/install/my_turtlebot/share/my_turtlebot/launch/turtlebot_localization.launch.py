from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(package="my_turtlebot", executable="lidar_icp", name="lidar_icp"),
            Node(
                package="my_turtlebot",
                executable="move_turtlebot",
                name="move_turtlebot",
            ),
        ]
    )
