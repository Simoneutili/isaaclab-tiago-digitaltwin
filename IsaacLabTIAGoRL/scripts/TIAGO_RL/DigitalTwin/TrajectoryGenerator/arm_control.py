#! /usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TaskPrioritization.task_priority import TPManager
import numpy as np
import time 
# import rospy
import numpy as np
from TaskPrioritization.Trajectories.trajectory import Trajectory
from TaskPrioritization.Trajectories.interpolation_method.Poly5 import Poly5
np.set_printoptions(linewidth=100, precision=4)
from TaskPrioritization.Bridges.ros2_controllers import Ros2ArmController
import rclpy
from rclpy.executors import SingleThreadedExecutor
import matplotlib.pyplot as plt

from threading import Thread

class TrajectoryPlanning(): 
    def __init__(self, tp_manager: TPManager):
        
        # Initialize ROS 2 node
        if not rclpy.ok():
            rclpy.init()
            
        self.joint_names_arm = ['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
        self.arm_controller = Ros2ArmController("joint_states", "velocity_publisher", '/arm_velocitiy_controller/commands', '/joint_states', self.joint_names_arm)
        
        self.executor = SingleThreadedExecutor()
        # Create an executor and give it both nodes
        self.executor.add_node(self.arm_controller.reader)
        self.executor.add_node(self.arm_controller.publisher)
        
        # ✅ Start the ROS executor in a background thread
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        self.tp_manager = tp_manager
        self.tp_manager.add_default_robot('tiago')
        
        self.wheel_pos = [0.0, 0.0]  # Initial positions of the wheels
        
        #self.executor.spin_once(timeout_sec=0.01)
        # Get the initial joint positions
        self.joint_pos = [self.wheel_pos, self.arm_controller.get_joint_positions()]
        
        self.robot = self.tp_manager.get_robot_kinematics()
        self.period = 0.2  # Duration of the trajectory in seconds
        self.traj = Trajectory(self.period)
        self.tp_manager.initialize(initial_joint_pos=self.joint_pos)
        initial = self.robot.get_arm_ee_poses()[0].copy()
        print(f"\033[92mInitial end-effector position: {initial}\033[0m")
        
        p_final = initial.copy()
        p_final[0] -= 0.1  # Move in x direction
        self.traj.poly5(p_i=initial, p_f=p_final, invert=True)  # Invert=True to reverse the trajectory direction
        
        self.arm_ctrl = self.tp_manager.add_task("SingleArmControl", 1)
        self.arm_ctrl.set_trajectory(trajectory=self.traj)
        self.arm_ctrl.use_base_joints(False)
        self.arm_ctrl.set_gains(Kp=np.array([40] * 6))
        
        # joint_traj_control = self.tp_manager.add_task("JointTrajectoryControl", 0)
        # self.executor.spin_once(timeout_sec=0.01)
        #initial_joint_pos = self.arm_controller.get_joint_positions()
        # print(f"\033[92mInitial joint positions: {initial_joint_pos}\033[0m")
        
        # list_of_traj = [Poly5(q_0=initial_joint_pos[i], q_dot_0=0, q_dot_dot_0=0, q_f=0.1, q_dot_f=0, q_dot_dot_f=0, period=period) for i in range(len(initial_joint_pos))]
        # list_of_traj = []# 

        # for i in range(len(initial_joint_pos)):
        #     list_of_traj.append(Poly5(q_0=initial_joint_pos[i], q_dot_0=0, q_dot_dot_0=0, q_f=0.5, q_dot_f=0, q_dot_dot_f=0, period=self.period))  # Arm joints# 

        # joint_traj_control.set_arm_trajectories(list_of_traj)
        # sing_avoid = tp.add_task("SingularityAvoidance", 3,)
        # sing_avoid.set_singularity_values_limits(min_sing_value=0.01, max_sing_value=0.1)
        # sing_avoid.set_gains(Kp=np.array([0.5], dtype=np.float64))

        # tp.activate_plotting()

        # joint_traj_control.activate()
        # arm_ctrl.deactivate()

        #joint_traj_control.deactivate()
        self.arm_ctrl.activate()
        self.current_time = 0.0
        
        # Data lists as instance attributes
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.vx_data = []
        self.vy_data = []
        self.vz_data = []
        self.x_data_a = []
        self.y_data_a = []
        self.z_data_a = []
   
    def control(self):
        
        print(f"\033[92mCurrent time: {self.current_time}\033[0m")
        print("delta_t:", self.tp_manager._delta_t)
        
        self.current_time += self.tp_manager._delta_t
        #self.executor.spin_once(timeout_sec=0.01)

        joint_pos = [self.wheel_pos, self.arm_controller.get_joint_positions()]
        print(f"\033[92mJoint positions: {joint_pos}\033[0m")

        cmd = self.tp_manager.execute(joint_pos=joint_pos)
        print(f"\033[92mCommand: {cmd}\033[0m")
        
        cmd[2] = 0  # giunto del torso
        
        # print(f"\033[92mCommand: {cmd}\033[0m")
        # print(f"\033[92mJoint positions: {arm_controller.get_joint_positions()}\033[0m")

        self.arm_controller.publish_velocities(cmd[3:10])

        # Get target and actual positions/velocities
        t_pos, t_vel = self.traj.interpolate_pos(self.current_time)
        a_pos = self.arm_ctrl.get_actual_pos()
# 
        # Append to instance lists
        self.time_data.append(self.current_time)
        self.x_data.append(t_pos[0])
        self.y_data.append(t_pos[1])
        self.z_data.append(t_pos[2])
        self.vx_data.append(t_vel[0])
        self.vy_data.append(t_vel[1])
        self.vz_data.append(t_vel[2])
        self.x_data_a.append(a_pos[0])
        self.y_data_a.append(a_pos[1])
        self.z_data_a.append(a_pos[2])
        
        return cmd[3:10]  # Return only the arm joint velocities
    
    def plot_trajectory(self):
        
        # === Position plot (target vs actual) ===
        fig1, axs1 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        # X
        axs1[0].plot(self.time_data, self.x_data,   'r-',  label='X target')
        axs1[0].plot(self.time_data, self.x_data_a, 'r--', label='X actual')
        axs1[0].set_ylabel('Position X (m)')
        axs1[0].legend(loc='upper right')
        axs1[0].grid(True)
        # Y
        axs1[1].plot(self.time_data, self.y_data, 'g-', label='Y target')
        axs1[1].plot(self.time_data, self.y_data_a, 'g--', label='Y actual')
        axs1[1].set_ylabel('Position Y (m)')
        axs1[1].legend(loc='upper right')
        axs1[1].grid(True)
        # Z
        axs1[2].plot(self.time_data, self.z_data, 'b-', label='Z target')
        axs1[2].plot(self.time_data, self.z_data_a, 'b--', label='Z actual')
        axs1[2].set_ylabel('Position Z (m)')
        axs1[2].set_xlabel('Time (s)')
        axs1[2].legend(loc='upper right')
        axs1[2].grid(True)

        fig1.suptitle('EE Position: Target vs Actual')
        plt.tight_layout()
        plt.show()

        # === Velocity plot ===
        fig2, axs2 = plt.subplots(3, 1, figsize=(8, 6))
        axs2[0].plot(self.time_data, self.vx_data, label='Vx', color='r')
        axs2[0].set_ylabel('Velocity X')
        axs2[1].plot(self.time_data, self.vy_data, label='Vy', color='g')
        axs2[1].set_ylabel('Velocity Y')
        axs2[2].plot(self.time_data, self.vz_data, label='Vz', color='b')
        axs2[2].set_ylabel('Velocity Z')
        axs2[2].set_xlabel('Time (s)')
        fig2.suptitle('Trajectory Velocities')
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        # Cleanup the executor and nodes
        self.executor.shutdown()
        self.arm_controller.reader.destroy_node()
        self.arm_controller.publisher.destroy_node()
        rclpy.shutdown()

    def get_velocities(self):
        return self.arm_controller.get_joint_velocities()