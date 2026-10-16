import os
from launch import LaunchDescription  # type: ignore
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess  # type: ignore
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution  # type: ignore
from launch_ros.actions import Node  # type: ignore


def generate_launch_description():
    map_name = LaunchConfiguration('map')
    map_yaml = PathJoinSubstitution(['/home/ubuntu/mcl_ws/maps', [map_name, '.yaml']])

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value='map4',
            description='Map to load from /home/ubuntu/mcl_ws/maps (map, map2, map3 or map4)'
        ),

        # ── Map Server ──────────────────────────────────────────────────
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_yaml,
                'use_sim_time': False
            }]
        ),

        # Configure and activate the lifecycle node after a short delay
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/map_server', 'configure'],
                    output='screen'
                )
            ]
        ),
        TimerAction(
            period=4.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/map_server', 'activate'],
                    output='screen'
                )
            ]
        ),

        # ── Simulated Robot ─────────────────────────────────────────────
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='mcl_robot',
                    executable='simple_robot',
                    name='simple_robot',
                    output='screen'
                )
            ]
        ),

        # ── Scan Bridge (publishes /odom_pose used by the MCL node) ─────
        TimerAction(
            period=4.5,
            actions=[
                ExecuteProcess(
                    cmd=['python3', '/home/ubuntu/mcl_ws/bridge/scan_bridge.py'],
                    output='screen'
                )
            ]
        ),

        # ── MCL Localizer ───────────────────────────────────────────────
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='mcl_localization',
                    executable='mcl_node',
                    name='mcl_node',
                    output='screen'
                )
            ]
        ),

        # ── RViz ────────────────────────────────────────────────────────
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    output='screen',
                    arguments=['-d', '/home/ubuntu/mcl_ws/rviz/mcl.rviz']
                )
            ]
        ),
    ])
