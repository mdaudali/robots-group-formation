import numpy as np
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion


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


class Controller(object):
    def __init__(self, name_space):
        self.name_space = name_space
        self._name = name
        self._velocity = np.zeros(2, dtype=np.float32)
        self._publisher = rospy.Publisher("/%s/cmd_vel" % name, Twist, queue_size=5)
        self._groundtruth = GroundtruthPose(name)

    def build_command(self, cmd):
        return "%s/%s" % (self.name_space, cmd)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vel):
        u, w = vel[0], vel[1]
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        self._velocity = vel
        self._publisher.publish(vel_msg)