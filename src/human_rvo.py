
from cairo_simulator.RVO import  RVO_update, reach, compute_V_des
from cairo_simulator.ground_vehicles import GroundVehicle, HuskyTank
from cairo_simulator.simulator import Simulator, SimObject
from cairo_simulator.log import Logger

import pybullet_data
import pybullet as p
import time
import os, sys
from math import sqrt, atan


START_POS = [[0, 5, 0 ], [0, 0, 0]]
START_POS2 = [[0, 5 ], [0, 0]]
END_POS = [[5, 3 ], [2, 6]]


def rvo_step():

    pass




if __name__ == "__main__":

    logger = Logger()

    sim = Simulator(logger=logger, use_ros=False, use_real_time=True) # Initialize the Simulator
    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])

    test_0 = HuskyTank("test_0", START_POS[0])
    test_1 = HuskyTank("test_1", START_POS[1])


    ws_model = dict()
    #robot radius
    ws_model['robot_radius'] = 0.6
    #circular obstacles, format [x,y,rad]
    # no obstacles
    ws_model['circular_obstacles'] = []
    # with obstacles
    # ws_model['circular_obstacles'] = [[-0.3, 2.5, 0.3], [1.5, 2.5, 0.3], [3.3, 2.5, 0.3], [5.1, 2.5, 0.3]]
    #rectangular boundary, format [x,y,width/2,heigth/2]
    ws_model['boundary'] = [] 




    X = START_POS2
    V = [[0, 0], [0, 0]]
    V_max = [10.0, 10.0]
    goal = END_POS

    
    try:
        while True:

            V_des = compute_V_des(X, goal, V_max)
            print(f"V_des{V_des}")

            V = RVO_update(X, V_des, V, ws_model)
            print(f"V{V}")

            theta = [atan(i[1]/(i[0]+.001)) for i in V]
            print(f"theta{theta}")

            vel = [sqrt(i[0]**2 + i[1]**2) for i in V]
            print(f"vel{vel}")

            test_0.set_vel_theta(vel[0], theta[0])
            test_1.set_vel_theta(vel[1], theta[1])

            sim.step()

            position, quaternion = p.getBasePositionAndOrientation(test_0.get_simulator_id())
            X[0] = position[0:2]
            print(f"x_pos{X}")

            position, quaternion = p.getBasePositionAndOrientation(test_1.get_simulator_id())
            X[1] = position[0:2]

            print()
            time.sleep(.1)
            i = 0
            """
            while i <100:
                sim.step()
                i += 1
            """


    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)