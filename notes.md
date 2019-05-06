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
[Habitat](https://github.com/facebookresearch/habitat-sim)(Facebook 2019, big gays) <br>

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


[On Evaluation of Embodied Navigation Agents](https://arxiv.org/pdf/1807.06757.pdf)(arxiv 2018) <br>

### Agent architecture
#### purely reactive sensory input 
(sensory input-> DNN -> action) <br>
[learning to act by predicting the future](https://arxiv.org/pdf/1611.01779.pdf)([code](https://github.com/IntelVCL/DirectFuturePrediction), ICLR 2017) <br>
[Target-driven visual navigation in indoor  scenes  using  deep  reinforcement  learning](https://arxiv.org/pdf/1609.05143.pdf)([code](https://github.com/yushu-liu/icra2017-visual-navigation), ICRA 2017) <br>
#### equip with short-term memory
[reinforcement learning with unsupervised auxiliary tasks](https://arxiv.org/pdf/1611.05397.pdf)([code](https://github.com/miyosuda/unreal), ICLR 2017) <br>
[Playing FPS Games with Deep Reinforcement Learning](https://arxiv.org/pdf/1609.05521.pdf)([code](https://github.com/glample/Arnold), AAAI 2017) <br>
[Learning to navigate in complex environments](https://arxiv.org/pdf/1611.03673.pdf)(ICLR 2017) <br>
[Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning](https://arxiv.org/pdf/1805.01956.pdf)([code](https://github.com/mfe7/cadrl_ros)(IROS 2018) <br>
#### internal representations
(use more advanced memory mechanisms that support the construction of rich internal representations of the agent's environment) <br>
[Control of memory,  active perception,  and action in Minecraft](https://web.eecs.umich.edu/~baveja/Papers/ICML2016.pdf)(ICML 2016) <br>
[Cognitive mapping and planning for visual navigation](https://arxiv.org/pdf/1702.03920.pdf)(CVPR 2017) <br>
[Unifying map and landmark based representations for visual navigation](https://arxiv.org/pdf/1712.08125.pdf)(arXiv 2017) <br>
[Neural map:  Structured memory for deep reinforcement learning](https://arxiv.org/pdf/1702.08360.pdf)(ICLR 2018) <br>
[Semi-parametric topological memory for navigation](https://arxiv.org/pdf/1803.00653.pdf)([code](https://github.com/nsavinov/SPTM), ICLR 2018) <br>

### Summary
[tensorforce](https://mp.weixin.qq.com/s?__biz=Mzg2MjExNjY5Mg==&mid=2247483685&idx=1&sn=c73822b5b719db40648700d4242499e3&chksm=ce0d8f1ef97a06082fd9032b42d8699bc19adce339f2435f5b6a61c0ea9beb1cfc926509a8e0&mpshare=1&scene=1&srcid=&pass_ticket=9Mwfi8nrJduWesFYZOvfaN1uXqSrd%2B2CuQl%2FzqbUNmBAfv%2Bx%2BxgJyw8xSQfYkcsl#rd)(Chinese) <br>
[this work](https://mp.weixin.qq.com/s?__biz=Mzg2MjExNjY5Mg==&mid=2247483714&idx=1&sn=449c6c1b00272d31b9093e8ae32e5ca5&chksm=ce0d8f79f97a066fcc5929cdbd0fc83ce8412eaf9d97a5c51ed16799d7e8a401027dc3bb6486&mpshare=1&scene=1&srcid=&pass_ticket=9Mwfi8nrJduWesFYZOvfaN1uXqSrd%2B2CuQl%2FzqbUNmBAfv%2Bx%2BxgJyw8xSQfYkcsl#rd)(Chinese) <br>
