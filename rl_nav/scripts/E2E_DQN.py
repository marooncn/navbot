from tensorforce.agents import DQNAgent
import os
import env
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

maze_id = config.maze_id
restore = False

GazeboMaze = env.GazeboMaze(maze_id=maze_id, continuous=False)

record_dir = 'record'
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

saver_dir = './models/nav{}'.format(maze_id)
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

summarizer_dir = './record/E2E_DQN/nav{}'.format(maze_id)
if not os.path.exists(summarizer_dir):
    os.makedirs(summarizer_dir)

states = dict(
    image=GazeboMaze.states,
    previous_act=dict(shape=(2,), type='float'),
    relative_pos=dict(shape=(2,), type='float')
)


network_spec = [
     [
          dict(type='input', names=['image']),
          dict(type='conv2d', size=32, window=(8, 6), stride=4, activation='relu', padding='SAME'),
          dict(type='conv2d', size=64, window=(4, 3), stride=2, activation='relu', padding='SAME'),
          dict(type='pool2d', pooling_type='max', window=2, stride=2, padding='SAME'),
          dict(type='conv2d', size=64, window=2, stride=2, activation='relu', padding='SAME'),
          dict(type='flatten'),
          dict(type='output', name='image_output')
     ],
     [
          dict(type='input', names=['previous_act']),
          dict(type='dense', size=8, activation='relu'),
          dict(type='output', name='action_output'),
     ],
     [
          dict(type='input', names=['relative_pos']),
          dict(type='dense', size=16, activation='relu'),
          dict(type='output', name='position_output'),
     ],

     [
          dict(type='input', names=['image_output', 'action_output', 'position_output'], aggregation_type='concat'),
          dict(type='dense', size=512, activation='relu'),
          dict(type='dense', size=512, activation='relu'),

     ]

]

memory = dict(
    type='replay',
    include_next_states=True,
    capacity=10000
)

exploration = dict(
    type='epsilon_decay',
    initial_epsilon=1.0,
    final_epsilon=0.1,
    timesteps=10000,
    start_timestep=0
)

update_model = dict(
    unit='timesteps',
    # 64 timesteps per update
    batch_size=64,
    # Every 64 timesteps
    frequency=64
)

optimizer = dict(
    type='adam',
    learning_rate=0.0001
)

# Instantiate a Tensorforce agent
agent = DQNAgent(
    states=states,
    actions=GazeboMaze.actions,
    network=network_spec,
    # update_mode=update_model,
    # memory=memory,
    actions_exploration=exploration,
    saver=dict(directory='./models', basename='E2E_DQN_model.ckpt', load=restore, seconds=6000),
    summarizer=dict(directory='./record/E2E_DQN', labels=["graph", "losses", "reward", "'entropy'"], seconds=6000),
    optimizer=optimizer,
    double_q_model=True
)


episode = 0
episode_rewards = []
successes = []
total_timestep = 0
max_timesteps = 1000
max_episodes = 50000


while True:
    observation = GazeboMaze.reset()
    observation = observation / 255.0  # normalize
    agent.reset()

    timestep = 0
    episode_reward = 0
    success = False

    while True:
        state = dict()
        state['image'] = observation,
        state['previous_act'] = GazeboMaze.vel_cmd,
        state['relative_pos'] = GazeboMaze.p,
        # state = dict(image=observation, previous_act=GazeboMaze.vel_cmd, relative_pos=GazeboMaze.p)
        # print(state)

        # Query the agent for its action decision
        action = agent.act(state)
        print(action)
        # Execute the decision and retrieve the current information
        observation, terminal, reward = GazeboMaze.execute(action)
        observation = observation / 255.0  # normalize
        # print(reward)
        # Pass feedback about performance (and termination) to the agent
        agent.observe(terminal=terminal, reward=reward)
        timestep += 1
        episode_reward += reward
        if terminal or timestep == max_timesteps:
            success = GazeboMaze.success
            break

    episode += 1
    total_timestep += timestep
    # avg_reward = float(episode_reward)/timestep
    successes.append(success)
    episode_rewards.append([episode_reward, timestep, success])

    # if total_timestep > 100000:
    #     print('{}th episode reward: {}'.format(episode, episode_reward))

    if episode % 100 == 0:
        f = open(record_dir + '/E2E_DQN_nav' + str(maze_id) + '.txt', 'a+')
        for i in episode_rewards:
            f.write(str(i))
            f.write('\n')
        f.close()
        episode_rewards = []
        agent.save_model('./models/')

    if len(successes) > 100:
        if sum(successes[-100:]) > 80:
            GazeboMaze.close()
            agent.save_model('./models/')
            f = open(record_dir + '/DQN_nav' + str(maze_id) + '.txt', 'a+')
            for i in episode_rewards:
                f.write(str(i))
                f.write('\n')
            f.close()
            print("Training End!")
            break
