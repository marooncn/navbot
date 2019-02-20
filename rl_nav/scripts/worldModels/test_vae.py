import numpy as np
import VAE
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
import config

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

vae = VAE.VAE()
vae.set_weights(config.vae_weight)

data = np.load('./record/observation100.npy')
data = np.array([item for episode in data for item in episode])

indices = np.random.choice(data.shape[0])
frame = data[indices] / 255.0
# print(frame)
plt.subplot(1, 2, 1)
test = frame[:, :, ::-1]  # convert bgr to rgb format
plt.imshow(test)  # The rgb values should be in the range [0 .. 1] for floats or [0 .. 255] for integers.
plt.title('Original')
# plt.show()

z = vae.get_vector(frame.reshape(1, 48, 64, 3))
print(z)
output = vae.get_output(frame.reshape(1, 48, 64, 3))
# print(output)
plt.subplot(1, 2, 2)
result = output.reshape(48, 64, 3)[:, :, ::-1]
plt.imshow(result)
plt.title('Output')
plt.show()
