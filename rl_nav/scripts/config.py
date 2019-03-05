import rospkg
path = rospkg.RosPack().get_path('rl_nav')

# Environment related

# image
input_dim = (48, 64, 3)
# reward parameter
r_arrive = 100
r_collision = -50  # -100
Cr = 100
# goal
goal_space = []
start_space = []
goal_space_nav0 = [[-2.8, -4.0]]
start_space_nav0 = [[1., -3.], [1., -4.], [0, -4.], [0, -1.8], [-1, -3], [-1.8, -1.9], [-1.7, -3.8]]
goal_space.append(goal_space_nav0)
start_space.append(start_space_nav0)
Cd = 0.02
# max linear velocity
v_max = 0.5  # m/s
# max angular velocity
w_max = 1.2  # rad/s

# VAE related
latent_vector_dim = 32
vae_weight = path + '/scripts/worldModels/models/vae_weights_loss_0.0019.h5'

# MDN-RNN related
rnn_weight = path + '/scripts/worldModels/models/rnn_weights_129.h5'
