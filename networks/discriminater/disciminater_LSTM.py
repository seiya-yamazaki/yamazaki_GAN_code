import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class Discriminator_LSTM(chainer.Chain):

    def __init__(self, n_input, n_hidden, n_output):
        super(Discriminator_LSTM, self).__init__()
        with self.init_scope():
            self.L1 = L.LSTM(n_input, n_hidden)
            self.L2 = L.Linear(n_hidden, n_output)
        self.L1.reset_state()

    def reset_state(self):
        self.L1.reset_state()

    def __call__(self, z):
        h1 = F.tanh(self.L1(z))
        h2 = F.sigmoid(self.L2(h1))

        return h2

