import pybullet as p
import time
import copy
import rospy
from cairo_simulator.Simulator import Simulator, SimObject, ASSETS_PATH
from cairo_simulator.Manipulators import Sawyer

def main():
    rospy.init_node("CAIRO_Sawyer_Simulator")
    use_real_time = True

    sim = Simulator() # Initialize the Simulator

    # Add a table and a Sawyer robot
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
    sawyer_robot = Sawyer("sawyer0", 0, 0, 0.8)

    sim_obj = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    sim_obj = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    sim_obj = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    sim_obj = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))

    start_pos = [0]*7
    sawyer_robot.move_to_joint_pos(start_pos)


    joint_config = sawyer_robot.solve_inverse_kinematics([0.9,0,1.5], [0,0,0,1])

    joint_config2 = copy.copy(joint_config) #sawyer_robot.solve_inverse_kinematics([0.7,0,1.5], [0,0,0,1])
    joint_config2[0:3] = [0,0.2,0]

    sawyer_robot.set_default_joint_velocity_pct(0.5)
    traj = ((1., joint_config), (2., joint_config2), (2.5, joint_config), (5, joint_config2))

    time.sleep(3)
    while sawyer_robot.check_if_at_position(start_pos, 0.5) is False:
        time.sleep(0.1)
        pass

    sawyer_robot.execute_trajectory(traj)


    # Loop until someone shuts us down
    while rospy.is_shutdown() is not True:
        sim.step()


if __name__ == "__main__":
    main()
