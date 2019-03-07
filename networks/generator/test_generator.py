from unittest import TestCase
from generator import Generator
import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, variable

class TestGenerator(TestCase):

    BACHSIZE = 8
    SONGSIZE = 8
    LEARNINGNUM = 1000

    model = Generator(n_input=1, n_output=1, n_hidden=32)

    #song = np.ones((BACHSIZE, SONGSIZE)).astype(np.float32) * 0.5

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    for k in xrange(LEARNINGNUM):
        fsound = np.ones((BACHSIZE, 1)).astype(np.float32) * 0.5
        ret_song = fsound

        loss = 0

        for i in xrange(SONGSIZE):
            ret_song = model(ret_song)
            loss += F.mean_squared_error(ret_song, fsound)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        print "step {} {}".format(k, loss)


