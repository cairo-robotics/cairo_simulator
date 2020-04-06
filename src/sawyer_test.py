import pybullet as p
import time
import rospy
from cairo_simulator.Simulator import Simulator, SimObject, ASSETS_PATH
from cairo_simulator.Manipulators import Sawyer

def main():
    rospy.init_node("CAIRO_Sawyer_Simulator")
    use_real_time = True

    sim = Simulator() # Initialize the Simulator

    # Add a table and a Sawyer robot
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (1, 0, .5), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
    sawyer_robot = Sawyer("sawyer0", 0, 0, 0.8)

    joint_config = sawyer_robot.solve_inverse_kinematics([0.9,0,1.5], [0,0,0,1])
    sawyer_robot.move_to_joint_pos(joint_config)

    # Auto-advance the simulation timer
    p.setRealTimeSimulation(use_real_time)

    # Loop until someone shuts us down
    while rospy.is_shutdown() is not True:
        if use_real_time is False:
            p.stepSimulation()
            time.sleep(1./240.)
        sim.publish_robot_states()
        sim.process_trajectory_queues()
    p.disconnect()


if __name__ == "__main__":
    main()
