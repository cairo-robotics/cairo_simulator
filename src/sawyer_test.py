import pybullet as p
import rospy
from cairo_simulator.Simulator import Simulator, SimObject, ASSETS_PATH
from cairo_simulator.Manipulators import Sawyer


def main():
    rospy.init_node("CAIRO_Sawyer_Simulator")
    sim = Simulator(use_real_time = True)  # Initialize the Simulator

    # Add a Sawyer robot and register SimObjects
    
    sawyer_robot = Sawyer("sawyer0", 0, 0, 0.8)


    SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0),
                      (0, 0, 1.5708))  # Table rotated 90deg along z-axis
    SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))

    joint_config = sawyer_robot.solve_inverse_kinematics([0.9, 0, 1.5], [0, 0, 0, 1])
    # sawyer_robot.move_to_joint_pos(joint_config)

    # Loop until someone shuts us down
    while rospy.is_shutdown() is not True:
        sim.step()
    p.disconnect()


if __name__ == "__main__":
    main()
