"""Loads the arm controllers of the digital twin into a running TIAGo simulation and starts the bridge."""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

ROS_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLLERS = {
    "arm_velocity_controller": "velocity_controllers/JointGroupVelocityController",
    "arm_position_controller": "position_controllers/JointGroupPositionController",
}


def generate_launch_description():
    params_file = os.path.join(ROS_DIR, "arm_controllers.yaml")
    spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-type", controller_type, "--param-file", params_file, "--inactive"],
        )
        for name, controller_type in CONTROLLERS.items()
    ]
    bridge = ExecuteProcess(cmd=["python3", os.path.join(ROS_DIR, "gazebo_bridge.py")], output="screen")
    return LaunchDescription([*spawners, bridge])
