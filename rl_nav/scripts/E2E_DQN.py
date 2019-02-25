from tensorforce.agents import DQNAgent
from tensorforce.core.networks import Network
import os
import env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GazeboMaze = env.GazeboMaze(maze_id=0, continuous=False)

dir_name = 'record'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
restore = False

states = dict(
    image=GazeboMaze.states,
    action=dict(shape=(2,), type='float'),
    goal=dict(shape=(2,), type='float')
)

AModule = True   # is Action Module is valid
GModule = True   # is Goal Module is valid


class E2ENetwork(Network):

    def tf_apply(self, x, internals, update, return_internals=False):
        import tensorflow as tf
        image = x['image']
        action = x['action']
        goal = x['goal']
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

        # CNN
        weights = tf.get_variable(name='W1', shape=(8, 6, 3, 32), initializer=initializer)
        out = tf.nn.conv2d(image, filter=weights, strides=(1, 4, 4, 1), padding='SAME')
        out = tf.nn.relu(out)

        weights = tf.get_variable(name='W2', shape=(4, 3, 32, 64), initializer=initializer)
        out = tf.nn.conv2d(out, filter=weights, strides=(1, 2, 2, 1), padding='SAME')
        out = tf.nn.relu(out)

        out = tf.nn.max_pool(out, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        out = tf.nn.relu(out)

        weights = tf.get_variable(name='W3', shape=(2, 2, 64, 64), initializer=initializer)
        out = tf.nn.conv2d(out, filter=weights, strides=(1, 2, 2, 1), padding='SAME')
        out = tf.nn.relu(out)

        out = tf.layers.flatten(out)

        if AModule:
            action = tf.layers.dense(inputs=action, units=32, activation=tf.nn.relu)
            action = tf.layers.dense(inputs=action, units=16, activation=tf.nn.relu)

        if GModule:
            goal = tf.layers.dense(inputs=goal, units=32, activation=tf.nn.relu)
            goal = tf.layers.dense(inputs=goal, units=16, activation=tf.nn.relu)

        # append action, goal
        out = tf.concat([out, action, goal], axis=-1)
        out = tf.layers.dense(inputs=out, units=64, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=out, units=32, activation=tf.nn.relu)

        if return_internals:
            return out, None
        else:
            return out


memory = dict(
    type='replay',
    include_next_states=True,
    capacity=10000
)

exploration = dict(
    type='epsilon_decay',
    initial_epsilon=1.0,
    final_epsilon=0.005,
    timesteps=100000,
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
    network=E2ENetwork,
    update_mode=update_model,
    memory=memory,
    actions_exploration=exploration,
    optimizer=optimizer,
    double_q_model=True
)

if restore:
    agent.restore_model('./models/')

episode = 0
episode_rewards = []


while True:
    observation = GazeboMaze.reset()
    observation = observation / 255.0  # normalize
    agent.reset()

    timestep = 0
    max_timesteps = 1000
    max_episodes = 100000
    episode_reward = 0
    success = False
    action = [0, 0]

    while True:
        states['image'] = observation
        states['action'] = action
        states['goal'] = GazeboMaze.goal
        # print(state)

        # Query the agent for its action decision
        action = agent.act(states)
        # print(action)
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
    print('{} episode total reward: {}'.format(episode, episode_reward))

    if episode % 1000 == 0:
        f = open(dir_name + '/DQN_episode' + str(episode) + '.txt', 'w')
        for i in episode_rewards:
            f.write(str(i))
            f.write('\n')
        f.close()
        agent.save_model('./models/')

    if episode == max_episodes:
        break
