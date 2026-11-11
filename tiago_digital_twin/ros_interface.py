"""ROS 2 link between the Isaac Lab digital twin and the reference robot (Gazebo or the real TIAGo)."""

import threading

import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

ARM_JOINT_NAMES = [f"arm_{i}_joint" for i in range(1, 8)]


class VelocityLink:
    """Publishes the reference and simulated arm velocities and receives the measured ones."""

    def __init__(self, real_velocity_topic: str):
        # Isaac Sim handles Ctrl+C; rclpy's own handler would only make the spin thread raise on shutdown
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node("isaaclab_digital_twin")
        self._reference_pub = self._node.create_publisher(JointState, "/reference_velocities", 10)
        self._sim_pub = self._node.create_publisher(JointState, "/isaaclab_velocities", 10)
        self._node.create_subscription(JointState, real_velocity_topic, self._on_real_velocity, 10)
        self._joint_index = {name: i for i, name in enumerate(ARM_JOINT_NAMES)}
        self.real_velocity = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float32)
        """Latest measured arm joint velocities [rad/s]; joints missing from the message read as zero."""
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

    def publish(self, reference: float, sim_velocity: np.ndarray) -> None:
        """Publish the commanded reference and the velocities of the simulated arm."""
        stamp = self._node.get_clock().now().to_msg()
        self._reference_pub.publish(self._joint_state(stamp, [reference] * len(ARM_JOINT_NAMES)))
        self._sim_pub.publish(self._joint_state(stamp, sim_velocity.tolist()))

    def close(self) -> None:
        rclpy.try_shutdown()

    @staticmethod
    def _joint_state(stamp, velocity: list[float]) -> JointState:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = ARM_JOINT_NAMES
        msg.velocity = velocity
        return msg

    def _on_real_velocity(self, msg: JointState) -> None:
        velocity = np.zeros_like(self.real_velocity)
        for name, value in zip(msg.name, msg.velocity):
            index = self._joint_index.get(name)
            if index is not None:
                velocity[index] = value
        self.real_velocity = velocity
