import tensorflow as tf
import tensorflow.contrib as tc

# kaiming initializer
def kaiming_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tc.layers.variance_scaling_initializer(factor=2., mode='FAN_IN', uniform=uniform, seed=seed, dtype=dtype)

# xavier initializer
def xavier_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tc.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=uniform, seed=seed, dtype=dtype)

# batch normalization and relu
def bn_relu(x, training): 
    return tf.nn.relu(tf.layers.batch_normalization(x, training=training))

# layer normalization and relu
def ln_relu(x):
    return tf.nn.relu(tc.layers.layer_norm(x))

def bn_activation(x, training, activation=None):
    x = tf.layers.batch_normalization(x, training=training)

    if activation:
        x = activation(x)

    return x

def ln_activation(x, activation=None):
    x = tc.layers.layer_norm(x)

    if activation:
        x = activation(x)

    return x

def norm_activation(x, normalization=None, activation=None, training=False):
    if normalization:
        x = normalization(x, training=training) if 'batch_normalization' in str(normalization) else normalization(x)
    if activation:
        x = activation(x)

    return x

def standard_normalization(images):
    mean, var = tf.nn.moments(images, [0, 1, 2])
    std = tf.sqrt(var)

    normalized_images = (images - mean) / std
    
    return normalized_images, mean, std

def range_normalization(images, normalizing=True):
    if normalizing:
        processed_images = tf.cast(images, tf.float32) / 128 - 1
    else:
        processed_images = tf.cast((tf.clip_by_value(images, -1, 1) + 1) * 128, tf.uint8)

    return processed_images

def log_normal(sample, mu, logsigma):
    """ Compute a Gaussian distribution density with mean mu and standard deviation exp(logsigma)
    
    Arguments:
        sample: x
        mu: mean
        logsigma: log of the standard deviation
    """
    log2pi = tf.constant([np.log(2 * np.pi)])           # log(2pi)
    inverse_sigma = tf.exp(-logsigma)                   # 1/sigma
    tmp = (sample - mu) * inverse_sigma                 # (x - mu)/sigma
    
    return -0.5 * (tmp**2 + 2 * logsigma + log2pi)      # log(N(x; mu, sigma**2))

def logsumexp(value, axis=None, keepdims=False):
    """ Compute the log of the sum of the exp of value
    
    Arguments:
    	value: tensor for which we compute logsumexp
    	axis: axis along which we apply sum
    	keepdims: if true, retains reduced dimensions with length 1. 
    """
    if axis is not None:
        max_value = tf.reduce_max(value, axis=axis, keepdims=True)
        value = value - max_value    # for numerical stability
        if keepdims == False:
            max_value = tf.squeeze(max_value)
    else:
        max_value = tf.reduce_max(value)
        value = value - max_value
        
    return max_value + tf.log(tf.reduce_sum(tf.exp(value),
                                            axis=axis, keepdims=keepdims))

def get_tensor(sess, name=None, op_name=None):
    if name is None and op_name is None:
        raise ValueError
    elif name:
        return sess.graph.get_tensor_by_name(name)
    else:
        return sess.graph.get_tensor_by_name(op_name + ':0')
