import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):

    def __init__(self, n_input, n_hidden, n_output):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.L1 = L.Linear(n_input, n_hidden)
            self.L2 = L.Linear(n_hidden, n_output)

    def __call__(self, z):
        h1 = F.tanh(self.L1(z))
        h2 = F.sigmoid(self.L2(h1))

        return h2

