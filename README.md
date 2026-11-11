![Digital Twin](./images/DigitalTwin.jpg)

# Real2Sim Digital Twin of the TIAGo Arm in Isaac Lab

Code of the paper *Online Real2Sim Calibration of Velocity-Controlled Digital Twins via Reinforcement Learning*
(IEEE/ASME AIM 2026, [doi:10.1109/AIM65483.2026.11658146](https://doi.org/10.1109/AIM65483.2026.11658146)).

A reinforcement learning agent calibrates an Isaac Lab simulation of the 7-joint TIAGo arm while it runs. The
simulated arm and a reference robot receive the same periodic joint velocity command; at every step the agent
sets the joint friction and the drive damping of the simulated arm so that its joint velocities match the
measured ones. The reference robot is the Gazebo TIAGo (Sim2Sim), or any robot that publishes its joint states
over ROS 2.

![Framework](./images/SimtoSimFramework.jpg)

## How it works

| | |
|---|---|
| Observation (35) | simulated velocity, reference velocity, measured velocity, absolute velocity error and active-joint mask, 7 values each |
| Action (28) | per joint: friction exponent, damping exponent, friction, damping. The first two set the upper bound `10^(3u)` of the range in which the last two pick the value |
| Reward | minus the mean absolute velocity error over the active joints |
| Reference | 2 Hz sine of amplitude 0.1 rad/s on every arm joint (`square` and `triangle` are available for evaluation) |
| Curriculum | starting from `initial_active_joints`, the next joint is added when the median cosine similarity between simulated and measured velocities over 200 steps exceeds 0.9 |
| Agents | SAC and PPO from Stable-Baselines3, with hyperparameters tuned with Ray Tune and Optuna |

ROS 2 topics (`sensor_msgs/JointState`, velocities of `arm_1_joint` ... `arm_7_joint`):

| Topic | Direction | Content |
|---|---|---|
| `/reference_velocities` | published | commanded reference |
| `/isaaclab_velocities` | published | velocities of the simulated arm (environment 0) |
| `/joint_states` | subscribed | velocities of the reference robot (`env.real_velocity_topic`) |

![Isaac Lab environments](./images/IsaacLabEnvs.jpg)

## Repository structure

```
tiago_digital_twin/        Isaac Lab task Isaac-DigitalTwin-TiAGO-v0
  tiago_env.py             environment and its configuration
  ros_interface.py         ROS 2 publishers and subscriber of the environment
  sb3_utils.py             Stable-Baselines3 wrapping shared by the scripts
  agents/                  SAC and PPO hyperparameters
  assets/tiago/            URDF and meshes of the robot (see NOTICE for their licenses)
scripts/
  train.py                 train or resume an agent
  play.py                  run a trained agent
  tune_cfg.py              hyperparameter sweeps for the Isaac Lab Ray tuner
ros/
  gazebo_bridge.launch.py  loads the arm controllers into the Gazebo TIAGo and starts the bridge
  gazebo_bridge.py         drives the Gazebo arm with the velocities of the digital twin
  arm_controllers.yaml     velocity and position controllers of the arm
```

## Installation

1. Install [Isaac Sim 4.2 and Isaac Lab 1.4.0](https://isaac-sim.github.io/IsaacLab/v1.4.0/source/setup/installation/index.html)
   and ROS 2 Humble.
2. In the Python environment of Isaac Lab, with ROS 2 sourced:

   ```bash
   source /opt/ros/humble/setup.bash
   pip install -e .
   ```

3. For the Gazebo reference robot, build the public
   [TIAGo simulation](https://github.com/pal-robotics/tiago_simulation) of PAL Robotics following its
   instructions. No change to the PAL packages is needed.

## Usage

Run the commands from the root of this repository, with ROS 2 sourced in every terminal.

### 1. Reference robot

```bash
ros2 launch tiago_gazebo tiago_gazebo.launch.py is_public_sim:=True
ros2 launch ros/gazebo_bridge.launch.py
```

The bridge forwards `/isaaclab_velocities` to the arm of the Gazebo robot. When no velocities arrive for a
second, it moves the arm back to the start pose of the Isaac Lab environment and holds it there.

### 2. Training

```bash
python scripts/train.py --num_envs 256 --headless                      # SAC
python scripts/train.py --algo ppo --num_envs 256 --headless           # PPO
python scripts/train.py --num_envs 256 --headless env.initial_active_joints=1   # with the joint curriculum
python scripts/train.py --num_envs 64 --headless --checkpoint logs/sb3/Isaac-DigitalTwin-TiAGO-v0/<run>/model.zip
```

Every setting of the environment (`env.`) and of the agent (`agent.`) can be overridden on the command line.
Runs are written to `logs/sb3/Isaac-DigitalTwin-TiAGO-v0/<date>_<time>/`: the model, the observation and
reward normalization statistics, the configuration, and the TensorBoard logs.

```bash
tensorboard --logdir logs/sb3/Isaac-DigitalTwin-TiAGO-v0
```

### 3. Running a trained agent

```bash
python scripts/play.py --real-time                                     # final model of the latest run
python scripts/play.py --real-time --checkpoint <run>/model.zip env.reference_shape=square
python scripts/play.py --real-time --algo ppo --checkpoint <ppo run>/model.zip
```

![Real-time plot](./images/RealTimePlot.jpg)

The reference, measured and simulated velocities can be compared live, for example with PlotJuggler.

### 4. Hyperparameter tuning

The sweeps in `scripts/tune_cfg.py` run with the Ray tuner of Isaac Lab. Use the tuner of Isaac Lab 2.0.0: the
one shipped with 1.4.0 closes the output of the training process, which stops the trial.

```bash
mkdir ray_tuner && cd ray_tuner
curl -LO https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.0.0/scripts/reinforcement_learning/ray/tuner.py
curl -LO https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.0.0/scripts/reinforcement_learning/ray/util.py

ray start --head
cd <IsaacLab>
./isaaclab.sh -p <this repo>/ray_tuner/tuner.py --run_mode local \
    --workflow <this repo>/scripts/train.py \
    --cfg_file <this repo>/scripts/tune_cfg.py --cfg_class TiagoSacJobCfg \
    --metric rollout/ep_rew_mean --num_samples 30 --repeat_run_count 1
ray stop
```

Each trial trains for the `n_timesteps` of the agent configuration. Copy the best hyperparameters into
`tiago_digital_twin/agents/`.

### Another reference robot

Send the robot the command published on `/reference_velocities` and set `env.real_velocity_topic` to the
`sensor_msgs/JointState` topic with its joint velocities.

## Tested configuration

| Software | Version |
|---|---|
| Ubuntu | 22.04 LTS |
| NVIDIA driver / CUDA | 550.144.03 / 12.4 |
| Python | 3.10 |
| Isaac Sim | 4.2.0 |
| Isaac Lab | 1.4.0 |
| Stable-Baselines3 | 2.3.2 |
| PyTorch | 2.4.0 |
| Gymnasium | 0.29.0 |
| Ray | 2.41.1 |
| ROS 2 | Humble |

## Citation

```bibtex
@inproceedings{utili2026real2sim,
  author    = {Utili, Simone and Govoni, Andrea and Palli, Gianluca},
  title     = {Online Real2Sim Calibration of Velocity-Controlled Digital Twins via Reinforcement Learning},
  booktitle = {2026 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM)},
  year      = {2026},
  doi       = {10.1109/AIM65483.2026.11658146}
}
```
