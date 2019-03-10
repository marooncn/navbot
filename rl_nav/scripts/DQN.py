from tensorforce.agents import DQNAgent
import os
import itertools

import config
import env
import worldModels.VAE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GazeboMaze = env.GazeboMaze(maze_id=1, continuous=False)

dir_name = 'record'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
restore = False

vae = worldModels.VAE.VAE()
vae.set_weights(config.vae_weight)


# Network as list of layers
network_spec = [
     dict(type='dense', size=512, activation='relu'),
     dict(type='dense', size=512, activation='relu'),
     dict(type='dense', size=512, activation='relu')]


memory = dict(
    type='replay',
    include_next_states=True,
    capacity=3000
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
    states=dict(shape=(36,), type='float'),  # GazeboMaze.states,
    actions=GazeboMaze.actions,
    network=network_spec,
    update_mode=update_model,
    memory=memory,
    actions_exploration=exploration,
    optimizer=optimizer,
    saver=dict(directory='./models', basename='DQN_model.ckpt', load=restore, seconds=600),
    summarizer=dict(directory='./record/DQN', labels=["graph", "losses", "reward", "entropy"], seconds=6000),
    double_q_model=True
)


episode = 0
episode_rewards = []


while True:
    observation = GazeboMaze.reset()
    observation = observation / 255.0  # normalize
    agent.reset()

    timestep = 0
    max_timesteps = 1000
    max_episodes = 10000
    episode_reward = 0
    success = False
    action = -1

    while True:
        latent_vector = vae.get_vector(observation.reshape(1, 48, 64, 3))
        latent_vector = list(itertools.chain(*latent_vector))  # [[ ]]  ->  [ ]
        relative_pos = GazeboMaze.p
        previous_act = GazeboMaze.vel_cmd
        state = latent_vector + relative_pos + previous_act

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
    # avg_reward = float(episode_reward)/timestep
    episode_rewards.append([episode_reward, timestep, success])
    # print('{} episode total reward: {}'.format(episode, episode_reward))

    if episode % 1000 == 0:
        f = open(dir_name + '/DQN_episode' + str(episode) + '.txt', 'w')
        for i in episode_rewards:
            f.write(str(i))
            f.write('\n')
        f.close()
        agent.save_model('./models/')

    if episode == max_episodes:
        break
