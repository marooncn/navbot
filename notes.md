### Basics
#### RL
My paper reading [notes](https://github.com/marooncn/learning_note/blob/master/paper%20reading/Reinforcement%20Learning.md) of Reinforcement Learning.
#### Simutation Environment
[gym-gazebo](https://github.com/erlerobot/gym-gazebo)(Erle Robotics 2016) <br>
[DeepMind Lab](https://github.com/deepmind/lab)(DeepMind 2016) <br>
[SUNGG](http://suncg.cs.princeton.edu/)(Princeton 2017) <br>
[Matterport3D](https://github.com/niessner/Matterport)(Princeton etc 2017) <br>
[AI2-THOR](https://github.com/allenai/ai2thor)(AI2 2017) <br>
[House3D](https://github.com/facebookresearch/House3D)(facebook 2018) <br>
[Gibson Env](https://github.com/StanfordVL/GibsonEnv)(Stanford 2018) <br>
##### Comparision 
<img alt="simulation framework" src="https://github.com/marooncn/learning_note/blob/master/paper%20reading/image/simulation%20framework.png"  width="500"> <br>
<img alt=" summary of popular environments" src="https://github.com/marooncn/learning_note/blob/master/paper%20reading/image/%20summary%20of%20popular%20environments.png"  width="500"> <br>

### Papers
[Building Generalizable Agents with a Realistic and Rich 3D Environment](https://arxiv.org/pdf/1801.02209.pdf)
<img alt="DDPG framework" src="https://github.com/marooncn/learning_note/blob/master/paper%20reading/image/img1_Building%20Generalizable%20Agents%20with%20a%20Realistic%20and%20Rich%203D%20Environment.jpg"  width="800"> <br>
* Environment <br>
House3D
* Success measure <br>
To declare success, we want to ensure that the agent
identifies the target room by its unique properties (e.g.  presence of appropriate objects in the room such as pan and knives for kitchen and bed for bedroom) instead of merely reaching there by luck. An episode is considered successful if both of the following two criteria are satisfied: (1) the agent
is  located  inside  the  target  room;  (2) the  agent  consecutively sees a  designated  object  category associated with that target room type for at least 2 time steps.  We assume that an agent sees an object if there are at least 4% of pixels in X belonging to that object.
* Reward <br>
-0.1(collision penalty) -0.1*timestep +10(success reward) <br>
original reward is too sparse, do reward shaping for each step: the difference of shortest distances between the agent's movement.
* result <br>
Our final gated-LSTM agent achieves a success rate of 35.8% on 50 unseen environments

