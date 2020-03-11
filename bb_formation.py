#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

import numpy as np

# region CONSTANTS
ROBOT_RADIUS = 0.105 / 2.
MAX_SPEED = 0.2
MAX_TURN = MAX_SPEED / ROBOT_RADIUS / 4.
# endregion CONSTANTS


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
    def __init__(self, name, max_speed=MAX_SPEED):
        self._name = name
        self._velocity = np.zeros(2, dtype=np.float32)
        self._publisher = rospy.Publisher("/%s/cmd_vel" % name, Twist, queue_size=5)
        self._groundtruth = GroundtruthPose(name)
        self._max_speed = max_speed

    def feedback_linearized(self, v, e=.2):
        u = v[0] * np.cos(self.pose[2]) + v[1] * np.sin(self.pose[2])
        w = (v[1] * np.cos(self.pose[2]) - v[0] * np.sin(self.pose[2])) / e
        #if u < 0.:
        #    if abs(w) < 1e-2:
        #        w += .02
        #    u = 0.
        #u = max(u, .05)
        w_abs = abs(w)
        u = max(u, 0.)
        if abs(w) > MAX_TURN:
            w = w / abs(w) * MAX_TURN
        return u, w

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity
        u, w = self.feedback_linearized(velocity)
        if u >= self._max_speed:
            u = self._max_speed
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        self._publisher.publish(vel_msg)

    @property
    def pose(self):
        if not self._groundtruth.ready:
            return None
        return self._groundtruth.pose

    @property
    def ready(self):
        return self._groundtruth.ready

    @property
    def speed(self):
        return np.linalg.norm(self._velocity)

    @property
    def radius(self):
        return ROBOT_RADIUS

    @property
    def name(self):
        return self._name


class ReferencePoint(object):
    def __init__(self):
        pass

    # pose of the reference point (x, y, yaw)
    @property
    def pose(self):
        return np.zeros(3, dtype=np.float32)

    # position of the reference point (x, y)
    @property
    def position(self):
        return np.zeros(2, dtype=np.float32)

    # heading of the reference point (x', y')
    @property
    def heading(self):
        return np.zeros(2, dtype=np.float32)


class RobotReferencePoint(ReferencePoint):
    def __init__(self, robot):
        super(ReferencePoint, self).__init__()
        self._robot = robot

    @property
    def pose(self):
        return self._robot.pose

    @property
    def position(self):
        return self._robot.pose[:2]

    @property
    def heading(self):
        yaw = self._robot.pose[2]
        d = np.array([np.cos(yaw), np.sin(yaw)])
        d /= np.linalg.norm(d)
        return d


class RobotAverageReferencePoint(ReferencePoint):
    def __init__(self, robots):
        super(ReferencePoint, self).__init__()
        self._robots = robots

    @property
    def pose(self):
        pose = np.zeros(3, dtype=np.float32)
        for robot in self._robots:
            pose += robot.pose
        pose /= len(self._robots)
        return pose

    @property
    def position(self):
        pos = np.zeros(2, dtype=np.float32)
        for robot in self._robots:
            pos += robot.pose[:2]
        pos /= len(self._robots)
        return pos

    @property
    def heading(self):
        yaw = self.pose[2]
        d = np.array([np.cos(yaw), np.sin(yaw)])
        d /= np.linalg.norm(d)
        return d


class Axis(object):
    def __init__(self, point, direction):
        self._point = point
        self._direction = direction / np.linalg.norm(direction)
        self._normal = np.array([-direction[1], direction[0]])

    def distance_from(self, point):
        return np.abs(np.dot(point - self._point, self._normal))

    def direction_from(self, point):
        ret = self._normal
        if self.on_left(point):
            ret *= -1
        return ret

    def on_left(self, point):
        return np.dot(point - self._point, self._normal) > 0.

    def on_right(self, point):
        return not self.on_left(point)


class Obstacle(object):
    def __init__(self):
        pass

    # get the distance from an obstacle to a point
    def distance_from(self, coord):
        return np.zeros(2, dtype=np.float32)

    # get the direction away from the obstacle at a point
    def direction_away(self, coord):
        return np.zeros(2, dtype=np.float32)

    # get the force exerted by the obstacle
    def get_force(self, coord):
        return np.zeros(2, dtype=np.float32)


class ObstacleCylinder(Obstacle):
    def __init__(self, centre, radius, influence):
        super(Obstacle, self).__init__()
        self._centre = centre
        self._radius = radius
        self._influence = influence

    def distance_from(self, coord):
        d = coord - self._centre
        m = np.linalg.norm(d)
        return m - self._radius

    def direction_away(self, coord):
        d = coord - self._centre
        m = np.linalg.norm(d)
        return d / m

    def get_force(self, coord):
        d = self.distance_from(coord)
        force = np.zeros(2, dtype=np.float32)
        if d <= self._influence:
            away = self.direction_away(coord)
            S = self._influence + self._radius + ROBOT_RADIUS
            R = self._radius + ROBOT_RADIUS
            d = d + self._radius - ROBOT_RADIUS
            O_magnitude = 0.
            if d <= S:
                O_magnitude = (S - d) / (S - R)
            if d <= R:
                O_magnitude = np.inf
            force = O_magnitude * away
        return force


class ObstacleRobot(Obstacle):
    def __init__(self, robot):
        super(Obstacle, self).__init__()
        self._robot = robot
        self._radius = 2 * ROBOT_RADIUS
        self._influence = 0.2

    def distance_from(self, coord):
        d = coord - self._robot.pose[:2]
        m = np.linalg.norm(d)
        return m - self._radius

    def direction_away(self, coord):
        d = coord - self._robot.pose[:2]
        m = np.linalg.norm(d)
        return d / m

    def get_force(self, coord):
        d = self.distance_from(coord)
        force = np.zeros(2, dtype=np.float32)
        if d <= self._influence:
            away = self.direction_away(coord)
            S = self._radius + self._influence
            R = self._radius
            O_magnitude = 0.
            if d <= S:
                O_magnitude = (S - d) / (S - R)
            if d <= R:
                O_magnitude = np.inf
            force = O_magnitude * away
        return force


class BBFormationControl(object):
    def __init__(self, reference, robots, offsets, obstacles):
        self._dead_radius = 0.15
        self._control_radius = 0.8
        self._reference = reference
        self._robots = robots
        self._offsets = offsets
        self._obstacles = obstacles
        self._persist = rospy.Time.now()
        self._noise = np.zeros(2, dtype=np.float32)
        pass

    def in_dead_zone(self, target, actual):
        d = target - actual
        m = np.linalg.norm(d)
        return m <= self._dead_radius

    def in_control_zone(self, target, actual):
        d = target - actual
        m = np.linalg.norm(d)
        return m <= self._control_radius

    def avoid_static_obstacle(self, robot):
        final_vec = np.zeros(2, dtype=np.float32)
        for obstacle in self._obstacles:
            final_vec += obstacle.get_force(robot.pose[:2])
        return final_vec

    def avoid_robot(self, me):
        final_vec = np.zeros(2, dtype=np.float32)
        for you in self._robots:
            if me == you:
                continue
            obstacle = ObstacleRobot(you)
            final_vec += obstacle.get_force(me.pose[:2])
        return final_vec

    def noise(self):
        n = rospy.Time.now()
        if n.nsecs - self._persist.nsecs > 5e8:
            self._persist = n
            self._noise = np.random.normal(0., .05, 2)
        return self._noise

    def maintain_formation(self, robot, offset):
        def maintain_formation_speed(target):
            delta_speed = 0.
            if not self.in_dead_zone(target, robot.pose[:2]):
                to_target = target - robot.pose[:2]
                heading = self._reference.heading
                if self.in_control_zone(target, robot.pose[:2]):
                    delta_speed = (np.linalg.norm(to_target) - self._dead_radius) / (
                                self._control_radius - self._dead_radius)
                else:
                    delta_speed = 1.
                if not np.dot(to_target, heading) > 0.:
                    delta_speed *= -1

            K = 1.
            return min(robot.speed + K * delta_speed, 2.)

        def maintain_formation_steer(target):
            K = MAX_SPEED / ROBOT_RADIUS / 16.
            H_desired = np.arctan2(self._reference.heading[1], self._reference.heading[0])
            if not self.in_dead_zone(target, robot.pose[:2]):
                heading = self._reference.heading
                axis = Axis(target, heading)
                to_axis = axis.direction_from(robot.pose[:2])
                heading_yaw = np.arctan2(heading[1], heading[0])
                to_axis_yaw = np.arctan2(to_axis[1], to_axis[0])
                if self.in_control_zone(target, robot.pose[:2]):
                    delta_steer = (np.linalg.norm(to_axis) - self._dead_radius) / (
                                self._control_radius - self._dead_radius)
                    H_desired = heading_yaw + K * delta_steer * (to_axis_yaw - heading_yaw)
                else:
                    H_desired = to_axis_yaw
            return H_desired

        target = self._reference.position + offset
        speed = maintain_formation_speed(target)
        yaw = maintain_formation_steer(target)
        d = np.array([np.cos(yaw), np.sin(yaw)])
        d /= np.linalg.norm(d)
        d *= speed
        return d
        #if not self.in_dead_zone(target, robot.pose[:2]):
        #    return (target - robot.pose[:2]) / np.linalg.norm(target - robot.pose[:2])
        #return np.zeros(2, dtype=np.float32)

    def move_to_goal(self, robot, goal):
        d = np.zeros(2, dtype=np.float32)
        if not self.in_dead_zone(goal, robot.pose[:2]):
            d = goal - robot.pose[:2]
            d /= np.linalg.norm(d)
        return d

    def update(self):
        for robot in self._robots:
            if not robot.ready:
                return

        goal = np.array([15., 0])

        if self.in_dead_zone(goal, self._reference.position):
            for robot in self._robots:
                robot.velocity = np.zeros(2, dtype=np.float32)
            return
        
        for i in range(len(self._robots)):
            robot = self._robots[i]
            offset = self._offsets[i]

            vec = np.zeros(2, dtype=np.float32)
            vec += 1.2 * self.avoid_robot(robot)
            vec += 2.0 * self.avoid_static_obstacle(robot)
            vec += 0.6 * self.move_to_goal(robot, goal + offset)
            vec += 1.2 * self.maintain_formation(robot, offset)
            #vec += self.noise()
            robot.velocity = vec



def run():
    rospy.init_node('bb_formation')
    robots = []
    for i in range(3):
        robots.append(Robot("tb3_%d" % i, MAX_SPEED if i != 0 else MAX_SPEED))

    reference = RobotAverageReferencePoint(robots)
    offsets = [np.array([-.2, -.6]), np.array([.2, .0]), np.array([-.2, .6])]
    obstacles = [ObstacleCylinder(np.array([6., 0.]), 1., .8), ObstacleCylinder(np.array([6., 3.]), 1., .8), ObstacleCylinder(np.array([6., -3.]), 1., .8),
                 ObstacleCylinder(np.array([9., -1.5]), 1., .8),ObstacleCylinder(np.array([9., 1.5]), 1., .8)]

    bb = BBFormationControl(reference, robots, offsets, obstacles)

    rate_limiter = rospy.Rate(20)
    while not rospy.is_shutdown():
        bb.update()
        rate_limiter.sleep()


if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        pass
