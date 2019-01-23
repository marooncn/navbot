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
import cv2
from cv_bridge import CvBridge, CvBridgeError

import config
import time
import env
from std_msgs.msg import Float32


uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
launch = roslaunch.parent.ROSLaunchParent(uuid, ['/home/maroon/catkin_ws/src/rl_nav/launch/nav_gazebo.launch'])
launch.start()
rospy.init_node('test')
time.sleep(10)


def set_start(x, y, theta):
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
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        result = set_state(state)
        assert result.success is True
        print("set the model state successfully")
    except rospy.ServiceException:
        print("/gazebo/get_model_state service call failed")


start = [0, 0]
set_start(start[0], start[1], math.pi)

print('reset simulation ...')
rospy.wait_for_service('/gazebo/reset_simulation')
try:
    # reset_proxy.call()
    rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
except rospy.ServiceException:
    print("/gazebo/reset_simulation service call failed")
print('reset simulation successfully')


rospy.wait_for_service('/gazebo/pause_physics')
try:
    # resp_pause = pause.call()
    rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    print("pause physics successfully")
except rospy.ServiceException:
    print("/gazebo/pause_physics service call failed")

time.sleep(5)
'''
rospy.wait_for_service('/gazebo/unpause_physics')
try:
    # resp_pause = pause.call()
    rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    print("unpause physics successfully")
except rospy.ServiceException:
    print("/gazebo/unpause_physics service call failed")
'''


get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
robot_state = None
rospy.wait_for_service('/gazebo/get_model_state')
try:
    get_state = get_state
    robot_state = get_state("robot", "world")  # "robot" relative to "world"
    assert robot_state.success is True
except rospy.ServiceException:
    print("/gazebo/get_model_state service call failed")
pos = robot_state.pose
print(pos)


print('capture image ...')
image_data = None
cv_image = None
while image_data is None:
    image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image)
    # h = image_data.height
    # w = image_data.width
    cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
    print(np.asarray(cv_image).reshape([-1, 64, 48, 3]))
print('capture image successfully')

print('get contact state')
contact_data = None
while contact_data is None:
    contact_data = rospy.wait_for_message('/contact_state', ContactsState, 5.0)
collision = contact_data.states != []
print(collision)

launch.shutdown()

'''
# Unpause simulation to make observation
print('unpause physics ...')
rospy.wait_for_service('/gazebo/unpause_physics')
try:
    # resp_pause = pause.call()
    # self.unpause()
    rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
except rospy.ServiceException:
    print("/gazebo/unpause_physics service call failed")
print('unpause physics successfully')

print('get contact state')
contact_data = None
while contact_data is None:
    contact_data = rospy.wait_for_message('/contact_state', ContactsState, 5.0)
collision = contact_data.states != []
print(collision)

print('capture image ...')
image_data = None
cv_image = None
while image_data is None:
    image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image)
    # h = image_data.height
    # w = image_data.width
    cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
print('capture image successfully')


print('pause physics ...')
rospy.wait_for_service('/gazebo/pause_physics')
try:
    # resp_pause = pause.call()
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
except rospy.ServiceException:
    print("/gazebo/pause_physics service call failed")
print('pause physics successfully')

'''




