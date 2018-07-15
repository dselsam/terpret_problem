import argparse
import sys
import tensorflow as tf
from options import parse_options
from util import SharedLogDirichletInitializer
from terpret_problem import TerpretProblem

if __name__ == "__main__":
    opts = parse_options()

    with tf.Session() as sess:
        tf.set_random_seed(opts.seed)

        tp = TerpretProblem(opts)
        tf.global_variables_initializer().run(session=sess)

        for epoch in range(opts.n_epochs):
            _, loss = sess.run([tp.update, tp.loss])
            sys.stdout.write("[%d] %.8f" % (epoch, loss))
            mus = sess.run(tp.mus)
            for i in range(opts.v):
                sys.stdout.write("%d " % round(100 * mus[i, 0]))
            sys.stdout.write("\n")
