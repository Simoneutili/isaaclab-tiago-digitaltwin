"""Isaac Lab environment in which an RL agent tunes the joint friction and damping of the TIAGo arm.

The arm tracks a periodic joint-velocity reference. At every step the agent sets the PhysX friction and
damping of the active arm joints and is rewarded for matching the joint velocities measured on the
reference robot (Gazebo or the real TIAGo), which are exchanged over ROS 2.
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import numpy as np
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

from .ros_interface import ARM_JOINT_NAMES, VelocityLink

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "tiago")
NUM_ARM_JOINTS = len(ARM_JOINT_NAMES)

# One period of each reference shape, as a function of the phase in [0, 1) with values in [-1, 1].
REFERENCE_SHAPES = {
    "sine": lambda phase: math.sin(2.0 * math.pi * phase),
    "square": lambda phase: 1.0 if 0.0 < phase < 0.5 else -1.0,
    "triangle": lambda phase: 4.0 * abs(phase - 0.5) - 1.0,
}

TIAGO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(ASSET_DIR, "tiago.urdf"),
        usd_dir=os.path.join(ASSET_DIR, "usd"),
        merge_fixed_joints=True,
        fix_base=True,
        make_instanceable=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"arm_[1-4]_joint": 0.5, "arm_[5-7]_joint": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(joint_names_expr=["arm_[1-7]_joint"], stiffness=0.0, damping=50.0),
        "body": ImplicitActuatorCfg(joint_names_expr=["(?!arm_[1-7]_joint).*"], stiffness=0.0, damping=500.0),
    },
)


@configclass
class TiagoEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 3.0
    # per joint: friction range exponent, damping range exponent, friction, damping
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4 * NUM_ARM_JOINTS,))
    # per joint: simulated velocity, reference velocity, real velocity, velocity error, active flag
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5 * NUM_ARM_JOINTS,))
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)
    robot_cfg: ArticulationCfg = TIAGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)

    real_velocity_topic = "/joint_states"
    """JointState topic with the velocities measured on the reference robot."""

    reference_shape = "sine"
    """Shape of the velocity reference: one of ``REFERENCE_SHAPES``."""
    reference_frequency = 2.0
    """Frequency of the velocity reference [Hz]."""
    reference_amplitude = 0.1
    """Amplitude of the velocity reference [rad/s]."""

    param_min = 1e-3
    """Lower bound of the tuned friction and damping."""
    param_max_exponent = 3.0
    """The agent picks the upper bound of each parameter as ``10**e`` with ``e`` in [0, param_max_exponent]."""

    initial_active_joints = NUM_ARM_JOINTS
    """Number of arm joints tuned from the start, counted from the shoulder. The curriculum adds the rest."""
    curriculum_window = 200
    """Number of steps over which simulated and real velocities are compared."""
    curriculum_threshold = 0.9
    """Median cosine similarity over the window above which the next joint becomes active."""

    error_threshold_start = 5.0
    """Velocity error [rad/s] that terminates an episode at its first step."""
    error_threshold_decay = 1e-3
    """Reduction of the termination threshold per episode step."""
    error_threshold_min = 0.01
    """Lower bound of the termination threshold [rad/s]."""


class TiagoEnv(DirectRLEnv):
    cfg: TiagoEnvCfg

    def __init__(self, cfg: TiagoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ros = VelocityLink(cfg.real_velocity_topic)
        self._arm_ids = self.robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)[0]
        self._reference_shape = REFERENCE_SHAPES[cfg.reference_shape]
        self._reference = 0.0

        self._active = torch.zeros(NUM_ARM_JOINTS, device=self.device)
        self._set_active_joints(cfg.initial_active_joints)
        self._error = torch.zeros(self.num_envs, NUM_ARM_JOINTS, device=self.device)
        self._real_vel = torch.zeros(NUM_ARM_JOINTS, device=self.device)

        window_shape = (self.num_envs, cfg.curriculum_window, NUM_ARM_JOINTS)
        self._sim_window = torch.zeros(window_shape, device=self.device)
        self._real_window = torch.zeros(window_shape, device=self.device)
        self._window_step = 0

    def close(self):
        self._ros.close()
        super().close()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["tiago"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        unit = (actions + 1.0) / 2.0
        upper = 10.0 ** (unit[:, : 2 * NUM_ARM_JOINTS] * self.cfg.param_max_exponent)
        params = unit[:, 2 * NUM_ARM_JOINTS :] * (upper - self.cfg.param_min) + self.cfg.param_min
        n = self._num_active
        self.robot.write_joint_friction_to_sim(params[:, :n], self._arm_ids[:n])
        self.robot.write_joint_damping_to_sim(params[:, NUM_ARM_JOINTS : NUM_ARM_JOINTS + n], self._arm_ids[:n])

    def _apply_action(self) -> None:
        phase = (self._sim_step_counter * self.physics_dt * self.cfg.reference_frequency) % 1.0
        self._reference = self.cfg.reference_amplitude * self._reference_shape(phase)
        self.robot.set_joint_velocity_target(
            torch.full((self.num_envs, NUM_ARM_JOINTS), self._reference, device=self.device), joint_ids=self._arm_ids
        )

        sim_vel = self.robot.data.joint_vel[:, self._arm_ids]
        self._ros.publish(self._reference, sim_vel[0].cpu().numpy())
        self._sim_window[:, self._window_step] = sim_vel
        self._real_window[:, self._window_step] = self._read_real_velocity()
        self._window_step += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._window_step == self.cfg.curriculum_window:
            self._update_curriculum()

        self._real_vel = self._read_real_velocity()
        self._error = (self._real_vel - self.robot.data.joint_vel[:, self._arm_ids]).abs() * self._active

        threshold = torch.clamp(
            self.cfg.error_threshold_start - self.cfg.error_threshold_decay * self.episode_length_buf,
            min=self.cfg.error_threshold_min,
        )
        terminated = torch.any(self._error > threshold.unsqueeze(1), dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        return -self._error.sum(dim=1) / self._num_active

    def _get_observations(self) -> dict:
        active = self._active.expand(self.num_envs, -1)
        obs = torch.cat(
            (
                self.robot.data.joint_vel[:, self._arm_ids] * active,
                self._reference * active,
                self._real_vel * active,
                self._error,
                active,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # spread out the resets so that the environments do not all restart at the same step
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # the base is fixed and the other joints are held still, so only the arm goes back to its start state
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids][:, self._arm_ids],
            self.robot.data.default_joint_vel[env_ids][:, self._arm_ids],
            self._arm_ids,
            env_ids,
        )

    def _read_real_velocity(self) -> torch.Tensor:
        return torch.from_numpy(self._ros.real_velocity).to(self.device)

    def _set_active_joints(self, count: int) -> None:
        self._num_active = count
        self._active[:count] = 1.0

    def _update_curriculum(self) -> None:
        """Activate the next joint once the simulated arm reproduces the real velocity profile."""
        n = self._num_active
        sim, real = self._sim_window[..., :n], self._real_window[..., :n]
        similarity = (sim * real).sum(dim=1) / (sim.norm(dim=1) * real.norm(dim=1) + 1e-6)
        if n < NUM_ARM_JOINTS and similarity.median() > self.cfg.curriculum_threshold:
            self._set_active_joints(n + 1)
        self._window_step = 0
