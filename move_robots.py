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

import time

X = 0
Y = 1
YAW = 2

# copied from obstacle_avoidance.py
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

def feedback_linearized(p, v, e):
	u = v[0] * np.cos(p[2]) + v[1] * np.sin(p[2])
	w = (v[1] * np.cos(p[2]) - v[0] * np.sin(p[2])) / e
	return np.array([u, w], dtype=np.float32)

class Robot(object):
	def __init__(self, name, radius):
		self._name      = name
		self._velocity  = np.zeros(2, dtype=np.float32)
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
		vel_msg.linear.x  = u
		vel_msg.angular.z = w
		self._vel = vel
		self._publisher.publish(vel_msg)

	@property
	def pose(self):
		if not self._groundtruth.ready:
			return None
		return self._groundtruth.pose

	@property
	def radius(self):
		return self._radius

	@property
	def name(self):
		return self._name

class Goal(object):
	def __init__(self, position):
		self._position = position

	@property
	def position(self):
		return self._position

class Obstacle(object):
	def __init__(self, position, radius, influence):
		self._position = position
		self._radius = radius
		self._influence = influence

	@property
	def position(self):
		return self._position

	@property
	def radius(self):
		return self._radius

	@property
	def influence(self):
		return self._influence

	def in_sphere_of_influence(self, position):
		d = position - self._position
		m = np.linalg.norm(d)
		return m - self._radius - self._influence < 0

class RobotObstacle(object):
	def __init__(self, robot):
		self._robot     = robot

	@property
	def position(self):
		return self._robot.pose[:2]

	@property
	def radius(self):
		return self._robot.radius

	@property
	def influence(self):
		return 0.2

	def in_sphere_of_influence(self, position):
		d = position - self.position
		m = np.linalg.norm(d)
		return m - self.radius - self.influence < 0
		
class RobotReferencePoint(object):
	def __init__(self, robot):
		self._robot = robot

	@property
	def speed(self):
		return self._robot.velocity[0]

	@property
	def heading(self):
		return self._robot.pose[2]

	@property
	def position(self):
		return self._robot.pose[:2]

	def formation_pose(self, other_pose):
		relative_pose = other_pose - self._robot.pose
		while relative_pose[YAW] < -np.pi:
			relative_pose[YAW] += 2 * np.pi
		while relative_pose[YAW] > np.pi:
			relative_pose[YAW] -= 2 * np.pi
		return relative_pose

MAX_SPEED   = 0.2
ROBOT_RADIUS = 0.1
MAX_TURN = MAX_SPEED / ROBOT_RADIUS
DEAD_RADIUS = 0.15
CONTROL_RADIUS = 0.6

class AxisLine(object):
	def __init__(self, point, direction):
		self._point = point
		self._direction = direction / np.linalg.norm(direction)

	def get_distance(self, point):
		dx = point - self._point
		y = np.dot(dx, self._direction) * self._direction - dx
		return np.linalg.norm(y)

	def in_dead_zone(self, point):
		return self.get_distance(point) <= DEAD_RADIUS

	def in_control_zone(self, point):
		return self.get_distance(point) <= CONTROL_RADIUS

	def on_left(self, point):
		a = self._point
		b = self._point + self._direction
		ab = b - a
		am = point - a
		return ab[0] * am[1] - ab[1] * am[0]

	def on_right(self, point):
		return not self.on_left(point)

class BBFormationControl(object):
	def __init__(self, robot, reference, desired, goal, obstacles, other_bots, leader):
		self._robot      = robot                     # the robot to be controlled
		self._reference  = reference                 # the reference from which formation position is determined
		self._desired    = desired                   # the desired formation position
		self._goal       = goal + desired            # the goal position
		self._obstacles  = obstacles                 # obstacles to be avoided
		self._other_bots = other_bots               # other robots
		self._leader     = leader
		if robot == leader:
			print("%s is the leader" % robot.name)	
		pass

	def in_dead_zone(self, aim):
		d = self._robot.pose[:2] - aim
		m = np.linalg.norm(d)
		return m <= DEAD_RADIUS

	def in_control_zone(self, aim):
		d = self._robot.pose[:2] - aim
		m = np.linalg.norm(d)
		return m <= CONTROL_RADIUS

	def control_speed(self, aim, gain):
		if self.in_dead_zone(aim):
			return 0.
		d = self._robot.pose[:2] - aim
		m = np.linalg.norm(d)
		if self.in_control_zone(aim):
			r1 = CONTROL_RADIUS - DEAD_RADIUS
			r2 = m - DEAD_RADIUS
			return gain * r2 / r1
		return gain

	def detect_formation_position(self):
		return self._reference.formation_pose(self._robot.pose)

	def move_to_goal(self):
		V_magnitude = 1.
		V_direction = self._goal - self._robot.pose[:2]
		V_direction = V_direction / np.linalg.norm(V_direction)
		return self.control_speed(self._goal, V_magnitude) * V_direction

	def avoid_static_obstacle(self):
		no_of_influencers = 0
		final_vec = np.zeros(2, dtype=np.float32)
		for obstacle in self._obstacles:
			if obstacle.in_sphere_of_influence(self._robot.pose[:2]):
				no_of_influencers += 1
				d_ = self._robot.pose[:2] - obstacle.position
				d  = np.linalg.norm(d_)
				S = obstacle.radius + ROBOT_RADIUS + obstacle.influence
				R = obstacle.radius + ROBOT_RADIUS
				O_magnitude = 0.
				if d <= S:
					O_magnitude = (S - d) / (S - R)
				if d <= R:
					O_magnitude = np.inf
				O_direction = d_ / np.linalg.norm(d_)
				final_vec += O_direction
		if no_of_influencers == 0:
			return np.zeros(2, np.float32)
		return final_vec / no_of_influencers

	def avoid_robot(self):
		if self._robot == self._leader:
			return np.zeros(2, dtype=np.float32)
		no_of_influencers = 0
		final_vec = np.zeros(2, dtype=np.float32)
		for bot in self._other_bots:
			if bot == self._robot:
				continue
			obstacle = RobotObstacle(bot)
			if obstacle.in_sphere_of_influence(self._robot.pose[:2]):
				no_of_influencers += 1
				d_ = self._robot.pose[:2] - obstacle.position
				d  = np.linalg.norm(d_)
				S = obstacle.radius + obstacle.influence
				R = obstacle.radius
				O_magnitude = 0.
				if d <= S:
					O_magnitude = (S - d) / (S - R)
				if d <= R:
					O_magnitude = np.inf
				O_direction = d_ / np.linalg.norm(d_)
				final_vec += O_direction
		if no_of_influencers == 0:
			return np.zeros(2, dtype=np.float32)
		ret = final_vec / no_of_influencers
		return ret

	def noise(self):
		return np.zeros(2, np.float32)

	def maintain_formation_speed(self):
		if self._robot == self._leader:
			return 0.
		R_mag = self._robot.velocity[0]
		F_pos = self._reference.position + self._desired
		K = 1.
		delta_speed = self.control_speed(F_pos, K)
		V_speed = R_mag + K * delta_speed
		return V_speed

	def maintain_formation_steer(self):
		if self._robot == self._leader:
			return 0.
		F_dir = self._reference.heading
		F_pos = self._reference.position + self._desired 
		R_pos = self._robot.pose[:2]
		R_dir = self._robot.pose[2]
		F_axis = AxisLine(F_pos, F_dir)
		
		delta_heading = None
		if F_axis.in_dead_zone(R_pos):
			delta_heading = 0.
		elif F_axis.in_control_zone(R_pos):
			m = F_axis.get_distance(R_pos)
			r = CONTROL_RADIUS - DEAD_RADIUS
			d = m - DEAD_RADIUS
			delta_heading = (np.pi / 2) * d / r
		else:
			delta_heading = np.pi / 2

		if F_axis.on_right(R_pos):
			delta_heading *= -1

		theta_diff = np.abs(R_dir - F_dir)
		if theta_diff > np.pi / 2:
			delta_heading *= 1

		H_desired = F_dir - delta_heading
		if self._reference.speed <= 1e-2:
			H_desi = F_pos - R_pos
			H_desired = np.arctan2(H_desi[1], H_desi[0])
		V_steer = H_desired - R_dir
		while V_steer < -np.pi:
			V_steer += 2 * np.pi
		while V_steer > np.pi:
			V_steer -= 2 * np.pi
		return 0.6 * V_steer

	def maintain_formation(self):
		vec = np.array([self.maintain_formation_speed(), self.maintain_formation_steer()])
		ret = np.array([vec[0] * np.cos(vec[1]), vec[0] * np.sin(vec[1])])
		return ret


	def turn_to(self, vel):
		u = np.linalg.norm(vel)
		w = np.arctan2(vel[1], vel[0])
		w_diff = w - self._robot.pose[2]
		while w_diff < -np.pi:
			w_diff += 2 * np.pi
		while w_diff > np.pi:
			w_diff -= 2 * np.pi
		return np.array([u, min(w_diff, MAX_TURN)])
		#return np.array([u, w_diff])
		

	def update(self):
		if self._robot.pose == None:
			print("robot pose not ready")
			return
		vec = np.zeros(2, dtype=np.float32)
		vec += 1.0 * self.avoid_static_obstacle()
		vec += 1.0 * self.avoid_robot()
		vec += 0.6 * self.move_to_goal()
		vec += 0.4 * self.maintain_formation()

		vec = self.turn_to(vec)

		vec[0] = min(vec[0], MAX_SPEED)

		if self._leader == self._robot:
			vec[0] = 0.18

		if self.in_dead_zone(self._goal):
			vec = np.zeros(2, dtype=np.float32)

		self._robot.velocity = vec

def rotate(coord, theta):
	c = np.cos(theta)
	s = np.sin(theta)
	r = np.array([[c, s],[-s, c]])
	return np.matmul(r, coord)

def camera(coord, camera_pose):
	p = coord - camera_pose[:2]
	return rotate(p, -camera_pose[2])

def run():
	NUM_OF_ROBOTS = 3
	rospy.init_node('move_robots')

	# setup list of turtle bots
	robots = []
	for i in range(NUM_OF_ROBOTS):
		robots.append(Robot("tb3_%d" % i, ROBOT_RADIUS))

	goal = np.array([10., -1.])
	reference = RobotReferencePoint(robots[0])
	form_control = []
	for i in range(len(robots)):
		robot = robots[i]
		desired = np.array([ 0., .6 * i])
		form_control.append(BBFormationControl(
			robot, reference, desired, goal,
			[Obstacle(np.array([6., 0.]), 1., .6)],
			robots, robots[0]))

	rate_limiter = rospy.Rate(20)
	while not rospy.is_shutdown():
		for cont in form_control:
			cont.update()
		rate_limiter.sleep()
	

	


if __name__ == "__main__":
	try:
		run()
	except rospy.ROSInterruptException:
		pass
