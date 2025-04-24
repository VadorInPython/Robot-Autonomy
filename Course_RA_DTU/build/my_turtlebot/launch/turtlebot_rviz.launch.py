import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('nav2_bringup')
    rviz_file_dir = get_package_share_directory('my_turtlebot')
    launch_dir = os.path.join(bringup_dir, 'launch')


    rviz_config_file = LaunchConfiguration('rviz_config_file')
    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config_file',
        default_value=os.path.join(
            rviz_file_dir, 'rviz', 'rviz_launch.rviz'
        ),
        description='Full path to the RVIZ config file to use'
    )

    rviz_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'rviz_launch.py')
        ),
        launch_arguments={'rviz_config': rviz_config_file}.items()
    )


    # Create the launch description and populate
    ld = LaunchDescription()
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(rviz_cmd)
    
    return ld
