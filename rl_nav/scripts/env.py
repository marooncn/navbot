from __future__ import absolute_import
from __future__ import print_function

import rospy
import roslaunch
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
# from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelStates

from tensorforce.environments import Environment
import numpy as np
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError

import config
# reward parameter
r_arrive = config.r_arrive
r_collision = config.r_collision
Cr = config.Cr   # judge arrival
Cd = config.Cd   # compute reward if no collision and arrival
v_max = config.v_max  # max linear velocity
w_max = config.w_max  # max angular velocity


class GazeboMaze(Environment):
    """
    Base environment class.
    """
    def __init__(self, maze_id=0, continuous=True):
        self.maze_id = maze_id
        self.continuous = continuous
        self.goal_space = config.goal_space[maze_id]
        self.start_space = config.start_space[maze_id]
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "nav_gazebo.launch maze_id:={}".format(self.maze_id))
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_state = rospy.ServiceProxy('/')
        self.reward_range = (-np.inf, np.inf)
        self.goal = self.goal_space[np.random.choice(len(self.goal_space))]
        start = self.start_space[np.random.choice(len(self.start_space))]

        d0, theta0 = self.rectangular2polar(self.goal[0]-start[0], self.goal[1]-start[1])
        self.p = [d0, theta0]  # relative target position

        self.img_rows = 64
        self.img_cols = 48
        self.img_channels = 3
        self._states = dict(shape=(self.img_rows, self.img_cols, self.img_channels), type='float')
        self._actions = dict(num_actions=3, type='int')
        if self.continuous:
            self._actions = dict(shape=(2, 1), min_value=-1, max_value=1, type='float')

    def __str__(self):
        raise 'GazeMaze ({})'.format(self.maze_id)

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        rospy.wait_for_service('/gazebo/pause_physics')

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        return None

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """

        # Resets the state of the environment and returns an initial observation.
        self.goal = self.goal_space[np.random.choice(len(self.goal_space))]
        start = self.start_space[np.random.choice(len(self.start_space))]
        d0, theta0 = self.rectangular2polar(self.goal[0] - start[0], self.goal[1] - start[1])
        self.p = [d0, theta0]  # relative target position
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        image_data = None
        success = False
        cv_image = None
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            # h = image_data.height
            # w = image_data.width
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        state = cv_image  # .reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state

    def execute(self, action):
        """
        Executes action, observes next state(s) and reward.

        Args:
            actions: Actions to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        if self.continuous:
            vel_cmd.linear.x = v_max*action[0]
            vel_cmd.angular.z = w_max*action[1]
        else:
            # 3 actions
            if action == 0:  # FORWARD
                vel_cmd.linear.x = 0.2
                vel_cmd.angular.z = 0.0
            elif action == 1:  # LEFT
                vel_cmd.linear.x = 0.05
                vel_cmd.angular.z = 0.2
            elif action == 2:  # RIGHT
                vel_cmd.linear.x = 0.05
                vel_cmd.angular.z = -0.2

        self.vel_pub.publish(vel_cmd)

        success = False
        done = False
        reward = 0

        contact_data = None
        while contact_data is None:
            contact_data = rospy.wait_for_message('/contact_state', ContactsState, timeout=5)
        collision = contact_data.states != []
        if collision:
            done = True
            reward = r_collision
        print(collision, contact_data.states)

        robot_data = None
        while robot_data is None:
            robot_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
        pos = robot_data.pose[2].position
        d_x = self.goal[0] - pos[0]
        d_y = self.goal[1] - pos[1]
        d, theta = self.rectangular2polar(d_x, d_y)
        if d < Cd:
            success = True
            done = True
            reward = r_arrive

        if not done:
            delta_d = self.p[0] - d
            reward = Cr*delta_d
        self.p = [d, theta]

        image_data = None
        cv_image = None
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        state = cv_image

        return state, reward, done, self.p

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are
        available simultaneously.

        Returns:
            States specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (default: 'float').
                - shape: integer, or list/tuple of integers (required).
        """
        return self._states

    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are
        available simultaneously.

        Returns:
            actions (spec, or dict of specs): Actions specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (required).
                - shape: integer, or list/tuple of integers (default: []).
                - num_actions: integer (required if type == 'int').
                - min_value and max_value: float (optional if type == 'float', default: none).
        """
        return self._actions

    def rectangular2polar(self, d_x, d_y):
        d = math.sqrt(d_x * d_x + d_y * d_y)
        theta = math.atan2(d_y, d_x)
        return d, theta
