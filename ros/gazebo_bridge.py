#!/usr/bin/env python3
"""Drives the arm of the Gazebo TIAGo with the velocity reference of the Isaac Lab digital twin.

The reference received on /reference_velocities is forwarded to the arm velocity controller, so the robot and
the digital twin track the same command. When none arrives for a second, the arm is switched to position
control and sent back to the start pose of the Isaac Lab environment.
"""

import math
import time

import rclpy
from controller_manager_msgs.srv import SwitchController
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

START_POSITION = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
VELOCITY_CONTROLLER = "arm_velocity_controller"
POSITION_CONTROLLER = "arm_position_controller"
# The trajectory controller of the PAL simulation claims the arm until it is deactivated.
ARM_CONTROLLERS = [VELOCITY_CONTROLLER, POSITION_CONTROLLER, "arm_controller"]
TIMEOUT_S = 1.0


class GazeboBridge(Node):
    def __init__(self):
        super().__init__("isaaclab_gazebo_bridge")
        self._velocity_pub = self.create_publisher(Float64MultiArray, f"/{VELOCITY_CONTROLLER}/commands", 10)
        self._position_pub = self.create_publisher(Float64MultiArray, f"/{POSITION_CONTROLLER}/commands", 10)
        self._switch_client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self._active_controller = None
        self._last_msg_time = time.monotonic()
        self.create_subscription(JointState, "/reference_velocities", self._on_reference, 10)
        self.create_timer(TIMEOUT_S, self._check_timeout)

    def _on_reference(self, msg: JointState) -> None:
        self._last_msg_time = time.monotonic()
        if any(math.isnan(v) for v in msg.velocity):
            self.get_logger().warn("Ignoring a velocity command with NaN values.")
            return
        self._activate(VELOCITY_CONTROLLER)
        self._velocity_pub.publish(Float64MultiArray(data=msg.velocity))

    def _check_timeout(self) -> None:
        if time.monotonic() - self._last_msg_time > TIMEOUT_S:
            self._activate(POSITION_CONTROLLER)
            self._position_pub.publish(Float64MultiArray(data=START_POSITION))

    def _activate(self, controller: str) -> None:
        if controller == self._active_controller or not self._switch_client.service_is_ready():
            return
        request = SwitchController.Request()
        request.activate_controllers = [controller]
        request.deactivate_controllers = [c for c in ARM_CONTROLLERS if c != controller]
        request.strictness = SwitchController.Request.BEST_EFFORT
        self._active_controller = controller
        self._switch_client.call_async(request).add_done_callback(self._on_switch_done)
        self.get_logger().info(f"Switching the arm to {controller}.")

    def _on_switch_done(self, future) -> None:
        if not future.result().ok:
            self.get_logger().error(f"Could not activate {self._active_controller}, retrying.")
            self._active_controller = None


def main():
    rclpy.init()
    rclpy.spin(GazeboBridge())


if __name__ == "__main__":
    main()
