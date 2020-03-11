from typing import List
from controller import Controller
from functools import partial
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

class GlobalController(object):
    def __init__(self, robots: List[Controller]):
        self.robots = robots
        for i in robots:
            rospy.Subscriber('/gazebo/model_states', ModelStates, partial(self.callback, i))
