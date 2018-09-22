def kl_loss(mu, logsigma, normal=True):
    if normal:
        return tf.reduce_mean(-0.5 * tf.reduce_sum(1. + 2. * logsigma - mu**2 - tf.exp(2 * logsigma), axis=1), axis=0)
    else:
        raise NotImplementedError
