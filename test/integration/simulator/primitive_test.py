import os
import sys
from functools import partial

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_simulator.core.link import get_joint_info_by_name

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import PRM


def main():
    config = {}
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        }
    ]
    config["primitives"] = [
        {
            "type": "box",
            "primitive_configs": {"w": .5, "l": .5, "h": .5},
            "sim_object_configs": 
                {
                    "object_name": "box",
                    "position": [1.0, -.3, .6],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "capsule",
            "primitive_configs": {"radius": .5, "height": .5},
            "sim_object_configs": 
                {
                    "object_name": "capsule",
                    "position": [2.0, -.3, .6],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "cylinder",
            "primitive_configs": {"radius": .5, "height": .5},
            "sim_object_configs": 
                {
                    "object_name": "cylinder",
                    "position": [3.0, -.3, .6],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "sphere",
            "primitive_configs": {"radius": .5},
            "sim_object_configs": 
                {
                    "object_name": "sphere",
                    "position": [4.0, -.3, .6],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },

    ]

    sim_context = SawyerSimContext(config)
    sim = sim_context.get_sim_instance()

    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
