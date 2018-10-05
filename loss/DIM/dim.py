import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tf_utils

def dim(feature_map, z, batch_size=None, log_tensorboard=False):
    with tf.variable_scope('loss'):
        with tf.variable_scope('local_MI'):
            E_joint, E_prod = _score(feature_map, z, batch_size)

            local_MI = E_joint - E_prod

        if log_tensorboard:
            tf.summary.scalar('E_joint_', E_joint)
            tf.summary.scalar('E_prod_', E_prod)
            tf.summary.scalar('Local_MI_', local_MI)
            
    return local_MI

def _score(feature_map, z, batch_size=None):
    with tf.variable_scope('discriminator'):
        T_joint = _get_score(feature_map, z, batch_size)
        T_prod = _get_score(feature_map, z, batch_size, shuffle=True)

        log2 = np.log(2.)
        E_joint = tf.reduce_mean(log2 - tf.math.softplus(-T_joint))
        E_prod = tf.reduce_mean(tf.math.softplus(-T_prod) + T_prod - log2)

    return E_joint, E_prod

def _get_score(feature_map, z, batch_size=None, shuffle=False):
    with tf.variable_scope('score'):
        height, width, channels = feature_map.shape.as_list()[1:]
        z_channels = z.shape.as_list()[-1]

        if shuffle:
            feature_map = tf.reshape(feature_map, (-1, height * width, channels))
            if batch_size is None:
                feature_map = tf.random.shuffle(feature_map)
            else:
                feature_map = _local_shuffle(feature_map, batch_size)
            feature_map = tf.reshape(feature_map, (-1, height, width, channels))
            feature_map = tf.stop_gradient(feature_map)
        
        # expand z
        z_padding = tf.tile(z, [1, height * width])
        z_padding = tf.reshape(z_padding, [-1, height, width, z_channels])
        
        feature_map = tf.concat([feature_map, z_padding], axis=-1)
        
        scores = _local_discriminator(feature_map, shuffle)
        scores = tf.reshape(scores, [-1, height * width])

    return scores

def _local_discriminator(feature_map, reuse):
    with tf.variable_scope('discriminator_net', reuse=reuse):
        x = tf.layers.conv2d(feature_map, 512, 1, kernel_initializer=tf_utils.kaiming_initializer())
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(feature_map, 512, 1, kernel_initializer=tf_utils.kaiming_initializer())
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(feature_map, 1, 1, kernel_initializer=tf_utils.kaiming_initializer())
        x = tf.nn.relu(x)

    return x

def _local_shuffle(x, batch_size):
    with tf.name_scope('local_shuffle'):
        _, d1, d2 = x.shape
        # d0 = tf.cond(self.is_training, lambda: tf.constant(self._args['batch_size']), lambda: tf.constant(1))
        d0 = batch_size
        b = tf.random_uniform(tf.stack([d0, d1]))
        idx = tc.framework.argsort(b, 0)
        idx = tf.reshape(idx, [-1])
        adx = tf.range(d1)
        adx = tf.tile(adx, [d0])

        x = tf.reshape(tf.gather_nd(x, tf.stack([idx, adx], axis=1)), (d0, d1, d2))

    return x
