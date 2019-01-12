from tensorforce.core.networks import Network
from tensorforce.agents import PPOAgent
import tensorflow as tf


class network(Network):

    def tf_apply(self, x):
        image = x['image']  # 640*480*3
        target = x['target']  # 2*1
        pre_vel = x['velocity']  # 2*1
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

        # CNN
        weights = tf.get_variable(name='W1', shape=(12, 12, 3, 32), initializer=initializer)
        image = tf.nn.conv2d(image, filter=weights, strides=(1, 4, 4, 1), padding='VALID')
        image = tf.nn.relu(image)

        weights = tf.get_variable(name='W2', shape=(10, 10, 32, 32), initializer=initializer)
        image = tf.nn.conv2d(image, filter=weights, strides=(1, 4, 4, 1), padding='VALID')
        image = tf.nn.relu(image)

        weights = tf.get_variable(name='W3', shape=(8, 7, 32, 16), initializer=initializer)
        image = tf.nn.conv2d(image, filter=weights, strides=(1, 2, 2, 1), padding='VALID')
        image = tf.nn.relu(image)

        image = tf.layers.flatten(image)
        image = tf.layers.dense(inputs=image, units=256, activation=tf.nn.relu)
        image = tf.layers.dense(inputs=image, units=128, activation=tf.nn.relu)

        target = tf.layers.dense(inputs=target, units=32, activation=tf.nn.relu)
        target = tf.layers.dense(inputs=target, units=16)

        pre_vel = tf.layers.dense(inputs=pre_vel, units=32, activation=tf.nn.relu)
        pre_vel = tf.layers.dense(inputs=pre_vel, units=16, activation=tf.nn.relu)

        state = tf.concat([image, target, pre_vel], 0)







