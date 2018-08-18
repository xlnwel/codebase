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
kaiming_initializer = tf.variance_scaling_initializer(scale=1/np.sqrt(2))
# xavier initializer
xavier_initializer = tf.variance_scaling_initializer()