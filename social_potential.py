#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
from functools import partial
import matplotlib.pyplot as plt
import argparse
import numpy as np
import rospy
from enum import Enum
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
import numpy as np


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
MAX_SPEED = 0.2
ROBOT_RADIUS = 0.1
NUM_OF_ROBOTS = 3
EPSILON = 0.2
X = 0
Y = 1
YAW = 2
FORMATION = Shape.COLUMN


# endregion


def obstacle_field(position, obstacle_positions, obstacle_radii):
    """

    :type obstacle_positions: List[Position]
    :type obstacle_radii: List[float]
    :type position: Position
    """
    v = np.zeros(2, dtype=np.float32)

    # Let d : Q x Q_obstacle -> R_>=0
    # be a distance function taking the configuration of the robot and the obstacle
    # where d(robot, obstacle) = 0 when the robot is touching the obstacle and positive otherwise
    # The Potential is then k/(2d^2(q, obstacle) + \eps) where K is a control gain and
    # eps is a small factor to prevent division by 0
    # Thus, the velocity is k / (d^3(q, obstacle) + \eps) * \delta d / \delta q
    # d is a naive distance measurement function, utilising euclidean distance
    # it assumes the robot is a single point defined by its control point
    # The direction of the force is given by \delta d / \delta q
    # As q is a vector, this is a vector with respect to a change in all parameters.
    # d is invariant to theta, so \delta d / \delta theta = 0 - There is no theta, ignore
    # We then bound the maximum force applied
    # We also bound the range of the bound using d_bound.

    d = (
        lambda vector_pos, obs_pos, obs_radii: np.linalg.norm(vector_pos - obs_pos)
                                               - obs_radii
    )
    k = 1.1
    d_bound = (
        2
    )

    def partial_diff(vector_pos, obs_pos, obs_radii):
        v_pos_x = np.add(vector_pos, np.array([0.05, 0]))
        v_pos_y = np.add(vector_pos, np.array([0, 0.05]))
        return normalize(
            np.array(
                [
                    d(v_pos_x, obs_pos, obs_radii) - d(vector_pos, obs_pos, obs_radii),
                    d(v_pos_y, obs_pos, obs_radii) - d(vector_pos, obs_pos, obs_radii),
                ]
            )
        )

    for obs_pos, obs_radii in zip(obstacle_positions, obstacle_radii):
        distance = d(position, obs_pos, obs_radii)
        if distance > d_bound:
            magnitude = 0
        elif distance <= 0.1:
            magnitude = 15
        else:
            magnitude = (2 - distance) / (2 - 0.1)

        v += normalize(position - obs_pos) * magnitude

    return v


def goal_potential_field(position, goal):
    """

    :type goal: Position
    :type position: Position
    """

    # ! Force field is described by some positive definite operator acting on q_goal - q
    v = np.zeros(2, dtype=np.float32)
    positive_definite_operator = partial(cap, max_speed=MAX_SPEED)
    v = goal - position
    control_gain = 0.7
    return v * control_gain


def attractive_potential_field(position, goal_pos, C, D):
    direction = normalize(goal_pos - position)
    r = np.linalg.norm(goal_pos - position)
    if r > C:
        magnitude = 1
    elif D < r <= C:
        magnitude = (r - D) / (C - D)
    else:
        magnitude = 0

    return direction * magnitude


def maintain_formation(position, robots, k):
    other_robots = [robot.pose for ind, robot in enumerate(robots) if ind != k]
    v = np.zeros(2, dtype=np.float32)
    attachment_points = []
    for pose in other_robots:
        attachment_point = np.array([pose[X] + FORMATION.value[0] * np.cos(pose[YAW]), pose[Y] + FORMATION.value[0] * np.sin(pose[YAW])])
        attachment_point = rotate(attachment_point, FORMATION.value[2])
        attachment_points.append(attachment_point)
        for iterations in range(1, FORMATION.value[1]):
            attachment_point = rotate(attachment_point, 2 * np.pi / FORMATION.value[1])
            attachment_points.append(attachment_point)

    min_point = np.array([np.inf, np.inf])
    for point in attachment_points:
        if np.linalg.norm(min_point - position) > np.linalg.norm(point - position):
            min_point = point
    return attractive_potential_field(position, min_point, 1.0, 0.0) * 1.3


def avoid_robots(position, robots, k):
    other_robots = [robot.pose[:2] for ind, robot in enumerate(robots) if ind != k]
    v = obstacle_field(position, other_robots, [0.1 for _ in range(len(robots) - 1)])
    control_gain = 1
    return v * control_gain


last_check = 0
last_value = 0

def random_velocity_field(position, shift=0.5, scaling_parameter=5):
    global last_check
    global last_value
    if time.time() > last_check + 2:
        last_check = time.time()
        last_value = np.random.uniform(0, 2 * np.pi)

    v = np.array([np.cos(last_value), np.sin(last_value)])
    control_gain = 0.1
    return v * control_gain


def move_to_unit_centre(position, robots, k):
    centre = np.mean(np.array([robot.pose[:2] for ind, robot in enumerate(robots) if ind != k]), axis=0)
    return attractive_potential_field(position, centre, 3, 2) * 0.6


def get_velocity(position, goal_position, obstacles, robots, k):
    v_goal = goal_potential_field(position, goal_position)
    v_obstacle = obstacle_field(position, *zip(*obstacles))
    v_robots = avoid_robots(position, robots, k)
    v_formation = maintain_formation(position, robots, k)
    v_unit = move_to_unit_centre(position, robots, k)
    v_random = random_velocity_field(position)
    v = v_obstacle + v_robots + v_goal + v_formation + v_unit + v_random
    return cap(v, max_speed=MAX_SPEED)


def feedback_linearized(pose, velocity, epsilon):
    u = 0.0  # [m/s]
    w = 0.0  # [rad/s] going counter-clockwise.
    # u = xdotp cos(theta) + ydotp sin(theta)
    # w = 1/eps (-xdotp sin(theta) + ydotp cos(theta))
    # velocity = xdot ydot

    u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
    w = (
            1
            / epsilon
            * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))
    )
    return u, w


def plot(robot, field):
    fig, ax = plt.subplots()
    # Plot field.
    X, Y = np.meshgrid(
        np.linspace(-20, 20, 30),
        np.linspace(-20, 20, 30),
    )
    U = np.zeros_like(X)
    V = np.zeros_like(X)

    for i in range(len(X)):
        for j in range(len(X[0])):
            velocity = field(np.array([X[i, j], Y[i, j]]))
            U[i, j] = velocity[0]
            V[i, j] = velocity[1]
    plt.quiver(X, Y, U, V, units="width")

    # Plot environment.
    ax.add_artist(plt.Circle([6., 0.], 1, color="gray"))
    ax.add_artist(plt.Circle(robot.pose[:2], .5, color="red"))
    # Plot a simple trajectory from the start position.
    # Uses Euler integration.
    # dt = 0.01
    # x = START_POSITION
    # positions = [x]
    # for t in np.arange(0.0, 40.0, dt):
    #     v = get_velocity(x, args.mode)
    #     x = x + v * dt
    #     positions.append(x)
    # positions = np.array(positions)
    # plt.plot(positions[:, 0], positions[:, 1], lw=2, c="r")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-0.5 - 20, 20 + 0.5])
    plt.ylim([-0.5 - 20, 20 + 0.5])
    plt.show()
    # plt.close()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, np.linalg.norm(np.concatenate((U, V)), axis=0), color='b')
    # plt.show()


def run():
    rospy.init_node('move_robots')

    # setup list of turtle bots
    robots = []
    for i in range(NUM_OF_ROBOTS):
        robots.append(Robot("tb3_%d" % i, ROBOT_RADIUS))

    goal = np.array([10., -1.])

    relative_formation = [
        # np.array([1, 0],  dtype=np.float32),
        # np.array([-1, 0],  dtype=np.float32),
        # np.array([0, 1], dtype=np.float32),
        np.array([0., 0.], dtype=np.float32),
        np.array([-2., 0.], dtype=np.float32),
        np.array([2., 0.], dtype=np.float32),
    ]
    print(relative_formation)
    obstacles = [(np.array([6., 0.]), 1.), (np.array([6., -3.]), 1.), (np.array([6., 3.]), 1.)]
    rate_limiter = rospy.Rate(20)

    while not rospy.is_shutdown():
        if any(r.pose is None for r in robots):
            rate_limiter.sleep()
            continue

        # expected_formations = create_expected_formations(relative_formation, robots)
        for k, robot in enumerate(robots):
            absolute_point_position = np.array(
                [
                    robot.pose[X] + EPSILON * np.cos(robot.pose[YAW]),
                    robot.pose[Y] + EPSILON * np.sin(robot.pose[YAW]),
                ],
                dtype=np.float32,
            )
            velocity = get_velocity(absolute_point_position, goal, obstacles, robots, k)
            # plot(robot, lambda x: get_velocity(x, goal, obstacles, robots, k))
            u, w = feedback_linearized(robot.pose, velocity, epsilon=EPSILON)
            robot.velocity = np.array([u, w])

        rate_limiter.sleep()


if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        pass
