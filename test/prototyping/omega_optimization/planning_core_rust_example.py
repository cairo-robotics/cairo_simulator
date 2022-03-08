from cmath import sqrt
from pathlib import Path

from cairo_planning_core import Agent


def main():

    settings_path = str(Path(__file__).parent.absolute()) + "/settings.yaml"
    rusty_sawyer_robot = Agent(settings_path, False, False)
 
   
    test_fk_config = [ 0.0, 0.0, -1.5708, 1.5708, 0.0, -1.5708, 0.0 ]
    print(test_fk_config)
    rusty_fk = rusty_sawyer_robot.forward_kinematics(test_fk_config)
    print(rusty_fk)

   
    collision_ik_results = []
    for _ in range(0, 10000):
        collision_ik_results = rusty_sawyer_robot.relaxed_inverse_kinematics(rusty_fk[0], rusty_fk[1]).data
        
    print(collision_ik_results)
    summation = 0
    for i in range(0, 7):
        summation += (test_fk_config[i] - collision_ik_results[i])**2
    print(sqrt(summation))
    
if __name__ == "__main__":
    main()
