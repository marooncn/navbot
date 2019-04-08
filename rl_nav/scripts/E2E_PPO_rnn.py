from tensorforce.agents import PPOAgent
import os
import env
import config

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
maze_id = config.maze_id
restore = False

GazeboMaze = env.GazeboMaze(maze_id=maze_id, continuous=True)

record_dir = 'record'
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

saver_dir = './models/nav{}'.format(maze_id)
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

summarizer_dir = './record/E2E_PPO_rnn/nav{}'.format(maze_id)
if not os.path.exists(summarizer_dir):
    os.makedirs(summarizer_dir)

states = dict(
    image=GazeboMaze.states,
    relative_pos=dict(shape=(2,), type='float'),
    previous_act=dict(shape=(2,), type='float'),
    previous_reward=dict(shape=(1,), type='float')
)


network_spec = [
     [
          dict(type='input', names=['image']),
          dict(type='conv2d', size=32, window=(8, 6), stride=4, activation='relu', padding='SAME'),
          dict(type='conv2d', size=64, window=(4, 3), stride=2, activation='relu', padding='SAME'),
          dict(type='pool2d', pooling_type='max', window=2, stride=2, padding='SAME'),
          dict(type='conv2d', size=64, window=2, stride=2, activation='relu', padding='SAME'),
          dict(type='flatten'),
          dict(type='dense', size=32, activation='relu'),
          dict(type='output', name='image_output')
     ],
     [
          dict(type='input', names=['image_output', 'relative_pos'], aggregation_type='concat'),
          dict(type='internal_lstm', size=256),
          dict(type='output', name='lstm_output'),
     ],

     [
          dict(type='input', names=['lstm_output', 'previous_act', 'previous_reward'], aggregation_type='concat'),
          dict(type='internal_lstm', size=256),
          dict(type='dense', size=256, activation='relu'),
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
    timesteps=100000,
    start_timestep=0
)

optimizer = dict(
    type='adam',
    learning_rate=0.0001
)

# Instantiate a Tensorforce agent
agent = PPOAgent(
    states=states,
    actions=GazeboMaze.actions,
    network=network_spec,
    # update_mode=update_model,
    # memory=memory,
    # actions_exploration=exploration,
    saver=dict(directory=saver_dir, basename='E2E_PPO_rnn_model.ckpt', load=restore, seconds=6000),
    summarizer=dict(directory=summarizer_dir, labels=["graph", "losses", "reward", "entropy"], seconds=6000),
    step_optimizer=optimizer,
)


episode = 0
test_episode = 0
total_timestep = 0
max_timesteps = 1000
max_episodes = 10000
episode_rewards = []
test_rewards = []
successes = []
test_successes = []

while True:
    agent.reset()
    observation = GazeboMaze.reset()
    observation = observation / 255.0  # normalize

    timestep = 0
    episode_reward = 0
    success = False

    if GazeboMaze.goal not in config.test_space[maze_id]:  # train
        while True:
            relative_pos = GazeboMaze.p
            previous_act = GazeboMaze.vel_cmd
            previous_reward = GazeboMaze.reward
            print(previous_act)
            state = dict(image=observation, relative_pos=relative_pos,
                         previous_act=previous_act, previous_reward=[previous_reward])
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
            f = open(record_dir + '/E2E_PPO_rnn_nav{}'.format(maze_id) + '.txt', 'a+')
            for i in episode_rewards:
                f.write(str(i))
                f.write('\n')
            f.close()
            episode_rewards = []
            agent.save_model('./models/')

    else:   # test
        while True:
            relative_pos = GazeboMaze.p
            previous_act = GazeboMaze.vel_cmd
            previous_reward = GazeboMaze.reward
            print(previous_act)
            state = dict(image=observation, relative_pos=relative_pos,
                         previous_act=previous_act, previous_reward=[previous_reward])
            # state = dict(image=observation, previous_act=GazeboMaze.vel_cmd, relative_pos=GazeboMaze.p)

            # Query the agent for its action decision
            action = agent.act(state, independent=True)
            # Execute the decision and retrieve the current information
            observation, terminal, reward = GazeboMaze.execute(action)
            observation = observation / 255.0  # normalize
            # print(reward)
            # Pass feedback about performance (and termination) to the agent
            # agent.observe(terminal=terminal, reward=reward)
            timestep += 1
            episode_reward += reward
            if terminal or timestep == max_timesteps:
                success = GazeboMaze.success
                break

        test_episode += 1
        test_successes.append(success)
        test_rewards.append([episode_reward, timestep, success])

        if test_episode % 100 == 0:
            f = open(record_dir + '/PPO_rnn_test_nav{}'.format(maze_id) + '.txt', 'a+')
            for i in test_rewards:
                f.write(str(i))
                f.write('\n')
            f.close()
            test_rewards = []

        if len(test_successes) > 100:
            if sum(test_successes[-100:]) > 50:
                GazeboMaze.close()
                agent.save_model('./models/')
                break
