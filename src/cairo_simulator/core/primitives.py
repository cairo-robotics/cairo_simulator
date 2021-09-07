import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

from collections import namedtuple

from cairo_simulator.core.simulator import SimObject

RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
BLUE = RGBA(0, 0, 1, 1)

def rpy2quat(rpy, degrees=False):
    """
    Produces the quaternion representation of euler angles in extrinsic RPY form.

    Args:
        rpy (array-like): The RPY vector
        degrees (bool, optional): False for radians, True for degrees. Defaults to False.

    Returns:
        ndarray: The quaternion in wxzy form.
    """
    quat = R.from_euler(
        'xyz', rpy, degrees=degrees).as_quat()
    return np.array((quat[3], quat[0], quat[1], quat[2]))

def unit_point():
    return [0., 0., 0.]

def unit_quat():
    return rpy2quat([0, 0, 0]) # [X,Y,Z,W]

def unit_pose():
    return [unit_point(), unit_quat()]

def get_box_geometry(width, length, height):
    return {
        'shapeType': p.GEOM_BOX,
        'halfExtents': [width/2., length/2., height/2.]
    }

def get_cylinder_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CYLINDER,
        'radius': radius,
        'length': height,
    }

def get_sphere_geometry(radius):
    return {
        'shapeType': p.GEOM_SPHERE,
        'radius': radius,
    }

def get_capsule_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CAPSULE,
        'radius': radius,
        'length': height,
    }

def create_collision_shape(geometry, pose=unit_pose(), client=0):
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        'physicsClientId': client,
    }
    collision_args.update(geometry)
    # if 'length' in collision_args:
    #     # TODO: pybullet bug visual => length, collision => height
    #     collision_args['height'] = collision_args['length']
    #     del collision_args['length']
    return p.createCollisionShape(**collision_args)

def create_body(collision_id=-1, visual_id=-1, mass=0, client=0):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=client)

def create_visual_shape(geometry, pose=unit_pose(), color=BLUE, specular=None, client=0):
    if (color is None): # or not has_gui():
        return -1
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        'physicsClientId': client,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)

def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = create_collision_shape(geometry, pose=pose) if collision else -1
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs) 
    return collision_id, visual_id

####################
# Primitive Shapes #
####################

def create_box(w, l, h, mass=0, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_box_geometry(w, l, h), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_cylinder(radius, height, mass=0, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_cylinder_geometry(radius, height), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_capsule(radius, height, mass=0, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_capsule_geometry(radius, height), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_sphere(radius, mass=0, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_sphere_geometry(radius), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

class PrimitiveBuilder():

    def __init__(self):

        self.primitive_type_fn_map = {
            'box': create_box,
            'cylinder': create_cylinder,
            'capsule': create_capsule,
            'sphere': create_sphere
        }
    
    def build(self, configs):
        primitive_fn = self.primitive_type_fn_map[configs['type']]
        body_id = primitive_fn(**configs['primitive_configs'])
        return SimObject(**configs['sim_object_configs'], model_file_or_sim_id=body_id)

    
