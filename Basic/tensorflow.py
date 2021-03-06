import tensorflow as tf

# to get a tensor's shape as a list
tensor.shape.as_list()


# to rename a tensor
logits = tf.identity(logits, name='logits')

def get_tensor(sess, name=None, op_name=None):
    if name is None and op_name is None:
        raise ValueError
    elif name:
        return sess.graph.get_tensor_by_name(name)
    else:
        return sess.graph.get_tensor_by_name(op_name + ':0')

# to build embedding table
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed

def tf_zip(x, y):
    return tf.stack([x, y], axis=1)

# kaiming initializer
def kaiming_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tf.contrib.layers.variance_scaling_initializer(factor=2, uniform=uniform, seed=seed, dtype=dtype)
# xavier initializer
def xavier_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tf.contrib.layers.variance_scaling_initializer(factor=1, uniform=uniform, seed=seed, dtype=dtype)

# relu and batch normalization
def bn_relu(layer, training): 
    return tf.nn.relu(tf.layers.batch_normalization(layer, training=is_training))

def add_weights_to_tensorboard(name_scope):
    with tf.variable_scope(name_scope, reuse=True):
        w = tf.get_variable('kernel')
        tf.summary.histogram('weights', w)

def add_gradients_to_tensorboard(name_scope):
    with tf.variable_scope(name_scope, reuse=True):
        grads = tf.gradients(loss, [tf.get_variable('conv2d/kernel'), tf.get_variable('conv2d_3/kernel'), tf.get_variable('conv2d_6/kernel')])
    grad_var_pairs = list(zip(grads, ['conv2d/kernel', 'conv2d_3/kernel', 'conv2d_6/kernel']))
    for grad, var in grad_var_pairs:
        tf.summary.histogram(var, grad)

def _variable_on_cpu(name, shape, initializer, dtype=tf.float32, cpu='/cpu:0'):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device(cpu):
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype)
    return var