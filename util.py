import tensorflow as tf

class SharedLogDirichletInitializer:
  def __init__(self, alpha, n_rows, n_cols):
    self.alpha = alpha
    self.n_rows = n_rows
    self.n_cols = n_cols
    self.dtype = tf.float32

  def __call__(self, shape, dtype=None, partition_info=None):
    dist = tf.distributions.Dirichlet([self.alpha for i in range(self.n_rows * self.n_cols)])
    return tf.reshape(tf.log(dist.sample([])), [self.n_rows, self.n_cols])

  def get_config(self):
    return { "alpha": self.alpha, "n_rows": self.n_rows, "n_cols": self.n_cols, "dtype": self.dtype.name }

class MaxEntInitializer:
  def __init__(self, n_rows, n_cols):
    self.n_rows = n_rows
    self.n_cols = n_cols
    self.dtype = tf.float32

  def __call__(self, shape, dtype=None, partition_info=None):
    return tf.ones(shape=[self.n_rows, self.n_cols], dtype=tf.float32)

  def get_config(self):
    return { "alpha": self.alpha, "n_rows": self.n_rows, "n_cols": self.n_cols, "dtype": self.dtype.name }
