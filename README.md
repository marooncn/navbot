# navbot
    It's a collection for mapless robot navigation using RGB image as visual input. It contains the test 
    environment and motion planners, aiming at realizing all the three levels of mapless navigation:
    1. memorizing efficiently; 
    2. from memorizing to reasoning; 
    3. more powerful reasoning
    The experiment data is in ./materials/record folder. 
    This work is under review of IEEE ITSC 2019.
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/mapless%20motion%20planner.png"  align='center' width="600"> <br>

## Environment 
<font face="Times New Roman">I built the environment as benchmark for testing the algorithms.</font> <br>

<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/environment.PNG"  align='center' width="600">    
<font face="Times New Roman">It has the following properties:</font> 

*  Diverse complexity. 
*  Gym-style Interface.
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/Interface.PNG" align='center' width="500"> 

*  Support ROS. 

Quickstart example code to use this benckmark.

    import env
    maze0 = env.GazeboMaze(maze_id=0, continuous=True)
    observation = maze0.reset()
    done = False
    while not done:
         # Stochastic strategy
         action = dict()
         action['linear_vel'] = np.random.uniform(0, 1)
         action['angular_vel'] = np.random.uniform(-1, 1)
         observation, done, reward = maze0.execute(action)
         print(action, reward)
    maze0.close()
    
## 1. Memorizing
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/E2E_PPO_nav2.gif" align='center' width="500"> 

### VAE-based planner
#### VAE Structure 
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/vae.png" align='center' width="200"> 

#### Train VAE 
Train in maze1 and maze2. The kl_tolerace is set to 0.5 (We stop optimizing for KL loss term once it is lower than some level, rather than letting it go to near zero) and latent dim is 32, thus the total loss is trained as close as possible to 16.
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/VAE_training.png" align='center' width="500"> <br> 
The following results are tested in maze3 to verify the ability of generalization.
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/vae_show.png" align='center' width="500"> 

#### Network Structure 
VAE-based planner & Baseline network structure  <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/VAE-DRL.png" align='center' width="300">  <img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/e2e.png" align='center' width="300"> 

### Performance

1. The proposed trajectory is blue and the baseline is green. <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/memorize_trajectory.PNG" align='center' width="300"> <br>
2. The success rate comparision in maze1. <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/result/maze1_dense_success.png" width="500"> 

3. Performance comparision <br>

 |   SPL   |   Benchmark |  Proposed |
 |---------|-------------|-----------|
 |  maze1  |    0.702    |   0.703   |
 |  maze2  |    0.611    |   0.626   |

That is, the proposed motion planner not only has much better sample-efficience, but also it has better performance. Actually, the shortest path in two mazes are both found by proposed motion planner (26 timesteps in maze1 and 29 time steps in maze2 with acceleration  in simulation).

## 2. From Memorizing to Reasoning
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/PPO_rnn_nav2.gif" align='center' width="500"> 

### Stacked LSTM and network structure
Stacked LSTM <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/stacked%20LSTM.png" width="500"> <br>
network structure <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/images/rnn.png" width="400"> 
### Result
Success rate in maze1 <br>
<img alt="" src="https://github.com/marooncn/navbot/blob/master/materials/result/maze1_rnn_success.png" width="500"> 

## Install
#### Ddependencies
tensorflow: 1.5.0 <br>
OS: Ubuntu 16.04  <br>
Python: 2.7 <br>
OpenCV: 3  <br>
ROS: Kinetic  <br>
Gazebo: 7   <br>
tensorforce: https://github.com/tensorforce/tensorforce  <br>

    # install tensorflow-gpu after cudnn and cuda are installed
    pip install tensorflow-gpu==1.5.0
    # or just use tensorflow-cpu if no Nvidia GPU, it can also work.
    pip install tensorflow==1.5.0
    # install OpenCV: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
    # install ROS: http://wiki.ros.org/kinetic/Installation/Ubuntu
    # install Gazebo 
    sudo apt-get install gazebo7 libgazebo7-dev
    # install tensorforce form source
    git clone https://github.com/tensorforce/tensorforce.git
    cd tensorforce
    sudo pip install -e . --user
    
#### Run
    sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
    sudo apt-get install ros-kinetic-turtlebot-*
    sudo apt-get remove ros-kinetic-turtlebot-description
    sudo apt-get install ros-kinetic-kobuki-description
    # change to catkin_ws/src
    git clone https://github.com/marooncn/navbot
    cd ..
    catkin_make
    source ./devel/setup.bash
    # you can change the configure in config.py
    cd src/navbot/rl_nav/scripts
    # run the proposed model for memorizing
    python PPO.py
    # run the proposed model for reasoning
    python E2E_PPO_rnn.py
#### Details
1. The default environment is maze1, you need to change maze_id in [nav_gazebo.launch](https://github.com/marooncn/navbot/blob/master/rl_nav/launch/nav_gazebo.launch) and [config.py](https://github.com/marooncn/navbot/blob/master/rl_nav/scripts/config.py) if you want change the environment. <br>
2. To execute [01_generate_data.py](https://github.com/marooncn/navbot/blob/master/rl_nav/scripts/worldModels/01_generate_data.py) to generate data, you need to comment the goal-related code in [nav_gazebo.launch](https://github.com/marooncn/navbot/blob/master/rl_nav/launch/nav_gazebo.launch) and [env.py](https://github.com/marooncn/navbot/blob/master/rl_nav/scripts/env.py). <br>
3. maze1 and maze2 are speeded up 10 times to train, if you want speed up other environments, just change 

       <max_step_size>0.001</max_step_size>
       <real_time_factor>1</real_time_factor>
   to
 
       <max_step_size>0.01</max_step_size>
       <!-- <real_time_factor>1</real_time_factor> -->
   in the environment file in [worlds](https://github.com/marooncn/navbot/tree/master/rl_nav/worlds).
4. To reproduce the result, please change the related parameters in [config.py](https://github.com/marooncn/navbot/blob/master/rl_nav/scripts/config.py) according to [config.txt](https://github.com/marooncn/navbot/blob/master/materials/record/config.txt).
5. PPO is not a deterministic policy gradient algorithm, the action at every timestep is sampled according to the distribution. It can be seen as "noise" and it's useful for explorations and generalizations. If you want to use the best strategy after the model is trained, just change 'deterministic = True' in [config.py](https://github.com/marooncn/navbot/blob/master/rl_nav/scripts/config.py) and the performance will be improved.

## Blog
[Introduction to tensorforce](https://mp.weixin.qq.com/s?__biz=Mzg2MjExNjY5Mg==&mid=2247483685&idx=1&sn=c73822b5b719db40648700d4242499e3&chksm=ce0d8f1ef97a06082fd9032b42d8699bc19adce339f2435f5b6a61c0ea9beb1cfc926509a8e0&mpshare=1&scene=1&srcid=&pass_ticket=9Mwfi8nrJduWesFYZOvfaN1uXqSrd%2B2CuQl%2FzqbUNmBAfv%2Bx%2BxgJyw8xSQfYkcsl#rd)(Chinese) <br>
[Introduction to this work](https://mp.weixin.qq.com/s?__biz=Mzg2MjExNjY5Mg==&mid=2247483714&idx=1&sn=449c6c1b00272d31b9093e8ae32e5ca5&chksm=ce0d8f79f97a066fcc5929cdbd0fc83ce8412eaf9d97a5c51ed16799d7e8a401027dc3bb6486&mpshare=1&scene=1&srcid=&pass_ticket=9Mwfi8nrJduWesFYZOvfaN1uXqSrd%2B2CuQl%2FzqbUNmBAfv%2Bx%2BxgJyw8xSQfYkcsl#rd)(Chinese) <br>

## Reference
[tensorforce](https://github.com/tensorforce/tensorforce)([blog](https://www.jiqizhixin.com/articles/2017-07-14-7?from=synced&keyword=tensorforce)) <br>
[gym_gazebo](https://github.com/erlerobot/gym-gazebo/blob/master/gym_gazebo/envs/turtlebot/gazebo_circuit2c_turtlebot_camera_nn.py) <br>
[gazebo](http://wiki.ros.org/gazebo) <br>
[roslaunch python API](http://wiki.ros.org/roslaunch/API%20Usage) <br>
[turtlebot_description](https://github.com/turtlebot/turtlebot/tree/kinetic/turtlebot_description) <br>
[kobuki_description](https://github.com/yujinrobot/kobuki/tree/devel/kobuki_description) <br>
[WorldModelsExperiments(official)](https://github.com/hardmaru/WorldModelsExperiments)  <br>
[WorldModels(by Applied Data Science)](https://github.com/AppliedDataSciencePartners/WorldModels) <br>
