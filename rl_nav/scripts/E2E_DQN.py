from tensorforce.agents import DQNAgent
import os
import env
import config

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

AModule = True   # is Action Module is valid
GModule = True   # is Goal Module is valid


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
          dict(type='input', names=['image_output', 'previous_act', 'relative_pos'], aggregation_type='concat'),
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
    type='epsilon_anneal',
    initial_epsilon=1.0,
    final_epsilon=0.2,
    timesteps=500000
)

update_model = dict(
    unit='timesteps',
    # 32 timesteps per update
    batch_size=32,
    # Every 4 timesteps
    frequency=4
)

optimizer = dict(
    type='adam',
    learning_rate=5e-3
)

# Instantiate a Tensorforce agent
agent = DQNAgent(
    states=states,
    actions=GazeboMaze.actions,
    network=network_spec,
    # update_mode=update_model,
    # memory=memory,
    actions_exploration=exploration,
    saver=dict(directory=saver_dir, basename='E2E_DQN_model.ckpt', load=restore, seconds=10800),
    summarizer=dict(directory=summarizer_dir, labels=["graph", "losses", "reward"], seconds=10800),
    optimizer=optimizer,
    # # target_sync_frequency=1000,  # Target network sync frequency
    double_q_model=True,
    huber_loss=1.0  # Huber loss clipping
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
        previous_act = GazeboMaze.vel_cmd
        print(previous_act)
        relative_pos = GazeboMaze.p
        state = dict(image=observation, previous_act=previous_act, relative_pos=relative_pos)
        # state = dict(image=observation, previous_act=GazeboMaze.vel_cmd, relative_pos=GazeboMaze.p)

        # Query the agent for its action decision
        action = agent.act(state)
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
    episode_rewards.append([episode_reward, timestep, success])

    # if total_timestep > 100000:
    #     print('{}th episode reward: {}'.format(episode, episode_reward))

    if episode % 1000 == 0:
        f = open(record_dir + '/E2E_DQN_episode' + str(episode) + '.txt', 'w')
        for i in episode_rewards:
            f.write(str(i))
            f.write('\n')
        f.close()
        episode_rewards = []
        agent.save_model('./models/')

    if len(successes) > 100:
        if sum(successes[-100:]) > 90:
            GazeboMaze.close()
            agent.save_model('./models/')
            break

    # if episode == max_episodes:
    #     break
