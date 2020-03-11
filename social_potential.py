#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
import yaml
from enum import Enum
from geometry_msgs.msg import Twist

# Occupancy grid.
from nav_msgs.msg import OccupancyGrid

# Position.
from tf import TransformListener

# Goal.
from geometry_msgs.msg import PoseStamped

# Path.
from nav_msgs.msg import Path


import rospy
from enum import Enum
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
import numpy as np

# For pose information.
from tf.transformations import euler_from_quaternion

# Import the potential_field.py code rather than copy-pasting.
import rrt

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, directory)
try:
    import rrt_navigation
except ImportError:
    raise ImportError(
        'Unable to import potential_field.py. Make sure this file is in "{}"'.format(
            directory
        )
    )
# region Utility functions


def rotate(vec, angle):
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(rotation_matrix, vec)


def cap(v, max_speed):
    n = np.linalg.norm(v)
    if n > max_speed:
        return v / n * max_speed
    return v


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-2:
        return np.zeros_like(v)
    return v / n


# endregion

# region Boilerplate classes
class Shape(Enum):
    SQUARE = (1.5, 4, 0)
    COLUMN = (1.5, 2, 0)
    DIAMOND = (1.5, 4, np.pi / 4)
    LINE = (1.5, 2, np.pi / 2)


class GroundtruthPose(object):
    def __init__(self, name='turtlebot3_burger'):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name

    def callback(self, msg):
        idx = [i for i, n in enumerate(msg.name) if n == self._name]
        if not idx:
            raise ValueError('Specified name "{}" does not exist.'.format(self._name))
        idx = idx[0]
        self._pose[0] = msg.pose[idx].position.x
        self._pose[1] = msg.pose[idx].position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose[idx].orientation.x,
            msg.pose[idx].orientation.y,
            msg.pose[idx].orientation.z,
            msg.pose[idx].orientation.w])
        self._pose[2] = yaw

    @property
    def ready(self):
        return not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose


class Robot(object):
    def __init__(self, name, radius):
        self._name = name
        self._velocity = np.zeros(2, dtype=np.float32)
        self._publisher = rospy.Publisher("/%s/cmd_vel" % name, Twist, queue_size=5)
        self._groundtruth = GroundtruthPose(name)
        self._radius = radius

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vel):
        u, w = vel[0], vel[1]
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        self._vel = vel
        self._publisher.publish(vel_msg)

    @property
    def pose(self):
        if not self._groundtruth.ready:
            return None
        return self._groundtruth.pose

    @property
    def yaw(self):
        return self.pose[1]

    @property
    def radius(self):
        return self._radius

    @property
    def name(self):
        return self._name


# endregion

# region Constants
MAX_SPEED = 0.5
ROBOT_RADIUS = 0.1
NUM_OF_ROBOTS = 3
EPSILON = 0.2
X = 0
Y = 1
YAW = 2
FORMATION = Shape.DIAMOND


# endregion


def check_ghosts(formation, occupancy_grid):
    return all(occupancy_grid.is_free(x) for x in formation)


def split(leader, sets):
    for i in sets:
        if leader in i:
            length = len(i)
            middle_index = length // 2

            first_half = i[:middle_index]
            second_half = i[middle_index:]

            if len(first_half):
                sets.append(first_half)
            if len(second_half):
                sets.append(second_half)
    return sets

def run():
    rospy.init_node('move_robots')
    with open('map.yaml') as fp:
        data = yaml.load(fp)
    img = rrt.read_pgm(data['image'])
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = rrt.UNKNOWN
    occupancy_grid[img < .1] = rrt.OCCUPIED
    occupancy_grid[img > .9] = rrt.FREE
    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]
    occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])
    # setup list of turtle bots
    robots = []
    for i in range(NUM_OF_ROBOTS):
        robots.append(Robot("tb3_%d" % i, ROBOT_RADIUS))

    goal = np.array([10., -1.])

    leaders = [[0, 1, 2]]

    rate_limiter = rospy.Rate(20)
    formations = [(0, 0), (0, 1), (0, 2)]
    while not rospy.is_shutdown():
        if any(r.pose is None for r in robots):
            rate_limiter.sleep()
            continue
        for i in leaders:
            leader = i[0]
            start_node, final_node = rrt.rrt(robots[leader].pose, goal + formations[leader], None)
            current_path = rrt_navigation.get_path(final_node)
            if not check_ghosts(current_path[1] + formations[i], occupancy_grid):
                split(leader, leaders)
                break
            else:
                for x in i:
                    robot = robots[x]
                    position = np.array(
                        [
                            robot.pose[X] + EPSILON * np.cos(robot.pose[YAW]),
                            robot.pose[Y] + EPSILON * np.sin(robot.pose[YAW]),
                        ],
                        dtype=np.float32,
                    )
                    start_node, final_node = rrt.rrt(robot.pose, current_path[1] + formations[x], occupancy_grid)
                    robot.current_path = rrt_navigation.get_path(final_node)
                    v = rrt_navigation.get_velocity(position, np.array(robot.current_path, dtype=np.float32))
                    u, w = rrt_navigation.feedback_linearized(robot.pose, v, epsilon=EPSILON)
                    robot.velocity = np.array([u, w])

        rate_limiter.sleep()

        
if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        pass
