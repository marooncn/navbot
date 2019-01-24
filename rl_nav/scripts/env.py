from __future__ import absolute_import
from __future__ import print_function

import rospy
import tf
import roslaunch
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
# from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState

from tensorforce.environments import Environment
import numpy as np
import math
import time
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
        # Launch the simulation with the given launch file name
        '''
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        cli_args = ['rl_nav', 'nav_gazebo.launch']  # , 'maze_id:={}'.format(self.maze_id)]
        roslaunch_args = cli_args[2:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args), roslaunch_args)]
        launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        launch.start()        
        '''

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, ['/home/maroon/catkin_ws/src/rl_nav/launch/nav_gazebo.launch'])
        self.launch.start()
        rospy.init_node('env_node')
        time.sleep(10)

        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        '''
        self.goal = self.goal_space[np.random.choice(len(self.goal_space))]
        start = self.start_space[np.random.choice(len(self.start_space))]
        self.set_start(start[0], start[1], np.random.uniform(0, 2*math.pi))
        d0, theta0 = self.rectangular2polar(self.goal[0]-start[0], self.goal[1]-start[1])
        self.p = [d0, theta0]  # relative target position        
        '''

        self.img_height = config.input_dim[0]
        self.img_width = config.input_dim[1]
        self.img_channels = config.input_dim[2]
        self._states = dict(shape=(self.img_height, self.img_width, self.img_channels), type='float')
        self._actions = dict(num_actions=3, type='int')
        if self.continuous:
            self._actions = dict(shape=(2, 1), min_value=-1, max_value=1, type='float')

    def __str__(self):
        raise 'GazeMaze ({})'.format(self.maze_id)

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.launch.shutdown()
        time.sleep(10)

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
        self.set_start(start[0], start[1], np.random.uniform(0, 2*math.pi))
        d0, theta0 = self.rectangular2polar(self.goal[0] - start[0], self.goal[1] - start[1])
        self.p = [d0, theta0]  # relative target position

        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # self.unpause
            rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        image_data = None
        cv_image = None
        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image)
            # h = image_data.height
            # w = image_data.width
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))  # width, height

        state = cv_image  # .reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state

    def execute(self, action):
        """
        Executes action, observes next state(s) and reward.

        Args:
            action: Actions to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause
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

        done = False
        reward = 0
        image_data = None
        cv_image = None

        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))
        state = cv_image

        contact_data = None
        while contact_data is None:
            contact_data = rospy.wait_for_message('/contact_state', ContactsState, timeout=5)
        collision = contact_data.states != []
        if collision:
            done = True
            reward = r_collision
        # print(collision, contact_data.states)

        robot_state = None
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            get_state = self.get_state   # create a handle for calling the service
            # use the handle just like a normal function, "robot" relative to "world"
            robot_state = get_state("robot", "world")
            assert robot_state.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

        pos = robot_state.pose.position
        d_x = self.goal[0] - pos.x
        d_y = self.goal[1] - pos.y
        d, theta = self.rectangular2polar(d_x, d_y)
        if d < Cd:
            done = True
            reward = r_arrive

        if not done:
            delta_d = self.p[0] - d
            reward = Cr*delta_d

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        self.p = [d, theta]

        return state, done, reward

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

    def set_start(self, x, y, theta):
        state = ModelState()
        state.model_name = 'robot'
        state.reference_frame = 'world'  # ''ground_plane'
        # pose
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        # twist
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")
