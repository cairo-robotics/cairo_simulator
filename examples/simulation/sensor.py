'''
Simple demo showing how to use a LaserRangeFinder sensor within CAIRO Sim.
Use the 1-4 keys to move the sensor around, check the terminal for readings from the sensor.
'''

import time
import os
import sys

if os.environ.get('ROS_DISTRO'):
    import rospy
import pybullet as p
import numpy as np
from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.log import Logger
from cairo_simulator.devices.sensors import LaserRangeFinder

def main():
    if os.environ.get('ROS_DISTRO'):
        rospy.init_node("CAIRO_Simulator_LRFSensorTest")
        use_ros = True
    else:
        use_ros = False

    use_real_time = True

    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros, use_real_time=use_real_time) # Initialize the Simulator
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    p.setPhysicsEngineParameter(enableFileCaching=0)

    # Add a few 0.05m x 0.05m x 0.05m cubes
    for idx in range(1,10):
        cube = SimObject(object_name="Cube%d"%idx, model_file_or_sim_id="cube_small.urdf", position=[0,0,idx/4.], orientation=[0,0,0,1])
        cube.move_to_pose([0,0,idx/4.],[0,0,0,1])
    # Add the sensor
    lrf_sensor = LaserRangeFinder(position_offset=[-0.5,0,0.3], fixed_pose=False)
    lrf_sensor.set_range(0.3,1.0) # Sensor takes readings from 0.3m to 1.0m away from its position.
    lrf_sensor.set_debug_mode(True)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
    z_inc_key = ord('1') # Press 1 to increase Z position of sensor
    z_dec_key = ord('2') # Press 2 to decrease Z position of sensor
    y_inc_key = ord('3') # Press 3 to increase Y position of sensor
    y_dec_key = ord('4') # Press 4 to decrease Y position of sensor

    loop_count = 0
    # Loop until someone shuts us down
    try:
        while True:

            # Control the sensor position
            keys = p.getKeyboardEvents()
            if z_inc_key in keys and keys[z_inc_key] & p.KEY_WAS_TRIGGERED:
                lrf_sensor._position_offset += np.array([0, 0, 0.05,])
            if z_dec_key in keys and keys[z_dec_key] & p.KEY_WAS_TRIGGERED:
                lrf_sensor._position_offset -= np.array([0, 0, 0.05,]) 
            if y_inc_key in keys and keys[y_inc_key] & p.KEY_WAS_TRIGGERED:
                lrf_sensor._position_offset += np.array([0, 0.05, 0,])
            if y_dec_key in keys and keys[y_dec_key] & p.KEY_WAS_TRIGGERED:
                lrf_sensor._position_offset -= np.array([0, 0.05, 0,]) 

            # Get and display a reading from the sensor
            if loop_count % 10 == 0:
                dist = lrf_sensor.get_reading()
                sim.logger.info("Detector Pos: %s \t\t\t Det: %f" % (lrf_sensor._position_offset, dist))
                loop_count = 0
            loop_count += 1            
            sim.step()
            time.sleep(1./60.)
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)
   



if __name__ == "__main__":
    main()
