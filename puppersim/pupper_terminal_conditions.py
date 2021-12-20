"""Contains the terminal conditions for locomotion tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import link

import gin
import numpy as np

from pybullet_envs.minitaur.envs_v2.utilities import minitaur_pose_utils
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils
from pybullet_envs.minitaur.envs_v2.utilities import termination_reason as tr


def get_ground_contacts(env):
    ground_contacts = []
    for ground_id in env._scene.ground_ids:
        ground_contacts += env._pybullet_client.getContactPoints(bodyA=ground_id)
    return ground_contacts


def has_robot_crashed(env):
    ground_contacts = get_ground_contacts(env)
    for contact in ground_contacts:
        robot_link_in_contact = contact[4]
        # print('link in contact: ', robot_link_in_contact)
        if (
            not robot_link_in_contact
            in env.robot._urdf_loader.get_end_effector_id_dict().values()
        ):
            return True
    return False


# link_dict = env.robot._urdf_loader.get_link_id_dict()
#   print(env.robot.robot_id, env.robot._urdf_loader.robot_id)
#   print(link_dict)
#   print(env._scene.ground_ids)
#   for contact in all_contacts:
#     body_a_id = contact[1]
#     body_b_id = contact[2]
#     link_a_id = contact[3]
#     link_b_id = contact[4]
#     print(body_a_id, body_b_id, link_a_id, link_b_id)
#     # if body_b_id ==
#   # raise Exception()
# return all_contacts


@gin.configurable
def default_terminal_condition_for_pupper(env):
    """A default terminal condition for Pupper.

    Pupper is considered as fallen if the base position is too low or the base
    tilts/rolls too much.

    Args:
      env: An instance of MinitaurGymEnv

    Returns:
      A boolean indicating if Minitaur is fallen.
    """
    roll, pitch, _ = env.robot.base_roll_pitch_yaw
    pos = env_utils.get_robot_base_position(env.robot)
    robot_crash = has_robot_crashed(env)
    # if robot_crash:
    #   print("ROBOT CRASHED")
    return (
        abs(roll) > 0.4
        or abs(pitch) > 0.4
        or pos[2] < 0.05
        or pos[2] > 0.6
        or robot_crash
    )
