A Python3, ROS-integrated, PyBullet-based Robot Simulator Foundation.


# Installation Instructions

### Install Python3 libraries for ROS (optional):
If you want to use Cairo Simulator with ROS, you need to enable Python3 ROS development and source the ROS environment scripts.

Using ROS is optional and the below steps are not strictly necessary.

```
sudo apt-get install python3-pip python3-yaml
pip3 install --user rospkg catkin_pkg
source /opt/ros/melodic/setup.bash
```

### Install the dependencies: 

#### If using ROS:

After sourcing the ROS environment scripts:
`pip3 install --user -r requirements.txt`

#### If using a Python virtual environment (i.e. virtualenv, pyenv etc,.):
`pip3 install pybullet -r requirements.txt`

or from the top level directory

`pip3 install -e ./`

# Running the Simulator
Unlike Gazebo, this is not a standalone executable program but rather is a foundation upon which a simulator can be created. The sawyer_test.py file contains a fairly minimal example showing how to use this package, initializing the simulator, a robot, and an object in the world.

# Simulator Components
The simulator uses 3 primary components: 
- Simulator: A singleton class that wraps the PyBullet physics simulation environment and keeps track of all simulation objects.
- Robot: A class that exposes common functionality relevant to robot control, including setting up ROS topics for joint position control and for broadcasting state.
- SimObject: A class that facilitates loading passive objects into the simulation, with functions for managing position and orientation.

## Robot Types
The initial version of this simulator only includes support for Manipulator arms, created by extending the `Manipulator` class. (See the Sawyer class in Manipulators.py for an example)

### Manipulator
Manipulator arms expose functionality to:
- `publish_state` (typically the configuration vector for all joint/gripper positions)
- `get_current_joint_states` which returns a configuration vector
- `move_to_joint_pos` which commands the arm to a given configuration
- `move_to_joint_pos_with_vel` which moves to a target configuration with per-joint velocity specifications
- `check_if_at_position` which returns whether the configuration of the manipulator is within some epsilon of a target configuration
- `execute_trajectory` which moves the arm through a sequence of configuration waypoints at specific timestamps.
- `solve_inverse_kinematics` which returns a configuration (list of joint positions) given a target position (and optional orientation) for the robot's end effector.
