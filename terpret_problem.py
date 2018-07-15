import argparse
import tensorflow as tf
import numpy as np
import sys
from util import SharedLogDirichletInitializer

class TerpretProblem:
    """Tensorflow encoding of the 'Terpret problem'"""
    def __init__(self, opts):
        self.opts = opts
        self.build_graph()

    def build_graph(self):
        opts = self.opts

        # mu0 : T [1, 2]
        # This is the marginal for the variable x0 which is observed to be 0.
        self.mu0 = tf.constant([[1.0, 0.0]], dtype=tf.float32)

        # ms : T [k-1, 2]
        # These are the parameters we are trying to optimize for, that control the marginal distribution
        # of the xs in the chain.
        self.ms = tf.get_variable("ms", shape=[opts.k - 1, 2], initializer=SharedLogDirichletInitializer(opts.alpha, opts.k - 1, 2))

        # mus : T [k, 2]
        # These are the parameters for the distributions of the full xs chain.
        self.mus = tf.concat([self.mu0, tf.nn.softmax(self.ms)], axis=0)

        # Returns the probability that `xor(xs[i], xs[i+1%k])` equals `0`
        def soft_xor_p0(i):
            j = tf.mod(i+1, opts.k)
            return (self.mus[i, 0] * self.mus[j, 0]) + (self.mus[i, 1] * self.mus[j, 1])

        # ys_eq_0 : T [k]
        self.ys_eq_0 = tf.map_fn(soft_xor_p0, tf.range(opts.k), dtype=tf.float32)
        self.loss = - tf.reduce_sum(tf.log(self.ys_eq_0))
        self.update = tf.train.AdamOptimizer(learning_rate=opts.learning_rate).minimize(self.loss)
