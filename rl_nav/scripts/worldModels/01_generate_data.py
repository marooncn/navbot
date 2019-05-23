import os
import sys
import argparse
import numpy as np

sys.path.append("..")
import env

dir_name = 'record'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def main(args):
    maze_id = args.maze_id
    GazeboMaze = env.GazeboMaze(maze_id=maze_id, continuous=True)
    print("Generating data for env maze{}".format(maze_id))
    file_number = args.file_number
    total_episodes = args.total_episodes
    # random_generated_int = np.random.randint(0, 2 ** 31 - 1)
    # np.random.seed(random_generated_int)
    time_steps = args.time_steps
    for i in range(file_number):
        total_frames = 0
        obs_data = []
        action_data = []
        episode = 0
        while episode < total_episodes:
            print('---------------------------')
            obs_sequence = []
            action_sequence = []

            observation = GazeboMaze.reset()
            done = False
            t = 0

            while t < time_steps:
                action = dict()
                action['linear_vel'] = np.random.uniform(0, 1)
                action['angular_vel'] = np.random.uniform(-1, 1)
                # print(action)
                obs_sequence.append(observation)
                action_sequence.append([action['linear_vel'], action['angular_vel']])
                t += 1

                observation, done, reward = GazeboMaze.execute(action)

            total_frames += t
            episode += 1
            obs_data.append(obs_sequence)
            action_data.append(action_sequence)
            print("{}th collections".format(i))
            print("Episode {} finished after {} timesteps".format(episode, t))
            print("Current dataset contains {} observations".format(total_frames))

        print("Saving dataset ...")
        np.save(dir_name+'/observation'+str(i), obs_data)
        np.save(dir_name+'/action'+str(i), action_data)
    GazeboMaze.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate new training data')
    parser.add_argument('--maze_id', type=int, default=1, help='which maze ')
    parser.add_argument('--file_number', type=int, default=300, help='total number of files to generate')
    parser.add_argument('--total_episodes', type=int, default=3000, help='how many episodes you need to record in a file')
    parser.add_argument('--time_steps', type=int, default=10, help='how many timesteps in an episode?')

    args = parser.parse_args()
    main(args)
