import tensorflow as tf
from tensorflow.keras import layers

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g = layers.Dense(in_channels // 8)
        self.theta = layers.Dense(in_channels // 8)
        self.phi = layers.Dense(in_channels // 8)
        self.W = layers.Dense(in_channels)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        out_channels = x.shape[-1]

        g_x = self.g(x)
        g_x = tf.reshape(g_x, (batch_size, out_channels // 8, 1))

        theta_x = self.theta(x)
        theta_x = tf.reshape(theta_x, (batch_size, out_channels // 8, 1))
        theta_x = tf.transpose(theta_x, perm=[0, 2, 1])

        phi_x = self.phi(x)
        phi_x = tf.reshape(phi_x, (batch_size, out_channels // 8, 1))

        f = tf.matmul(phi_x, theta_x)
        f_div_C = tf.nn.softmax(f, axis=-1)

        y = tf.matmul(f_div_C, g_x)
        y = tf.reshape(y, (batch_size, out_channels // 8))
        W_y = self.W(y)
        z = W_y + x

        return z