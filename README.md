# navbot
    It's a collection for mapless robot navigation using RGB image as visual input. It contains the test 
    environment, end-to-end network and the proposed motion planner.
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/mapless%20motion%20planner.png"  align='center' width="600"> <br>

## Environment 
<font face="Times New Roman">I built the environment for testing the algorithms.</font> <br>

<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/environment.PNG"  align='center' width="600">    
<font face="Times New Roman">It has the following properties:</font> 

*  Diverse complexity. 
*  Gym-style Interface.
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/Interface.PNG" align='center' width="500"> 

*  ROS-supporting.

## Memorize
### VAE
#### Structure 

<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/vae.png" align='center' width="200"> 

#### Result 

<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/vae_show.png" align='center' width="500"> 

### VAE-based Proposed Planner Compared with  benchmark 
The proposed is blue trajectory and the benchmark is green. <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/memorize_trajectory.PNG" align='center' width="300"> <br>
The reward comparision in maze1. <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/result/maze1_dense_reward.png" width="500"> 

## From Memorizing to Reasoning




## Install
    sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
    sudo apt-get install ros-kinetic-turtlebot-*
    sudo apt-get remove ros-kinetic-turtlebot-description
    sudo apt-get install ros-kinetic-kobuki-description
    # change to catkin_ws/src
    git clone https://github.com/marooncn/navbot
    cd ..
    catkin_make
    source ./devel/setup.bash
    # such as run PPO, you can change the configure in config.py
    cd src/navbot/rl_nav/scripts
    python PPO.py

## Reference
[WorldModelsExperiments(official)](https://github.com/hardmaru/WorldModelsExperiments)  <br>
[WorldModels(by Applied Data Science)](https://github.com/AppliedDataSciencePartners/WorldModels) <br>
[tensorforce](https://github.com/tensorforce/tensorforce)([blog](https://www.jiqizhixin.com/articles/2017-07-14-7?from=synced&keyword=tensorforce)) <br>
[gym_gazebo](https://github.com/erlerobot/gym-gazebo/blob/master/gym_gazebo/envs/turtlebot/gazebo_circuit2c_turtlebot_camera_nn.py) <br>
[gazebo](http://wiki.ros.org/gazebo) <br>
[roslaunch python API](http://wiki.ros.org/roslaunch/API%20Usage) <br>
[turtlebot_description](https://github.com/turtlebot/turtlebot/tree/kinetic/turtlebot_description) <br>
[kobuki_description](https://github.com/yujinrobot/kobuki/tree/devel/kobuki_description) <br>
