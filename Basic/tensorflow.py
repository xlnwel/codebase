import tensorflow as tf

# to get a tensor's shape as a list
tensor.shape.as_list()


# to rename a tensor
logits = tf.identity(logits, name='logits')


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