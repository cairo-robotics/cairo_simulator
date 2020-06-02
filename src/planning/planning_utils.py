from collections import namedtuple

JointInfo = namedtuple('JointInfo', ['idx', 'name', 'type',
                                     'qidx', 'uidx', 'flags',
                                     'damping', 'friction', 'lower_limit', 'upper_limit',
                                     'max_force', 'max_velocity', 'link_name', 'joint_axis',
                                     'parent_position', 'parent_orientation', 'parent_idx'])