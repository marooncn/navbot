# reward parameter
r_arrive = 100
r_collision = -100
Cr = 10
# goal
goal_space = []
start_space = []
goal_space_nav0 = [[-2.8, -4.0]]
goal_space.append(goal_space_nav0)
start_space_nav0 = [[1., -3.], [1., -4.], [0, -4.], [0, -1.8], [-1, -3], [-1.8, -3.5], [-2., -4.], [-2.6, -3.]]
start_space.append(start_space_nav0)
Cd = 0.02
# max linear velocity
v_max = 0.5  # m/s
# max angular velocity
w_max = 1  # rad/s


def adjust_obs(obs):
    return obs.astype('float32') / 255.
