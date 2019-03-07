from unittest import TestCase
from disciminater import Discriminator
import numpy as np

import chainer
import chainer.functions as F
from chainer import optimizers, variable
import random


class TestDiscriminator(TestCase):
    np.random.seed(0526)
    random.seed(0526)

    BACHSIZE = 8
    SONGSIZE = 8
    LEARNINGNUM = 1000
    INPUTNUM = 32

    model = Discriminator(n_input=INPUTNUM, n_output=1, n_hidden=32)


    optimizer = optimizers.Adam()
    optimizer.setup(model)

    true_song0 = np.ones((1, INPUTNUM)).astype(np.float32) * -0.5
    true_song1 = np.ones((1, INPUTNUM)).astype(np.float32) * 0.5

    song0 = np.ones((BACHSIZE / 2, INPUTNUM + 1)).astype(np.float32) * -0.5
    song1 = np.ones((BACHSIZE / 2, INPUTNUM + 1)).astype(np.float32) * 0.5

    song0[:, -1] = -1.0
    song1[:, -1] = 1.0

    song = np.r_[song0, song1]

    np.random.shuffle(song)
    np.random.shuffle(song)

    for k in xrange(1, LEARNINGNUM):

        loss = 0

        x = model(song[:,:INPUTNUM])

        loss = F.mean_squared_error(x, song[:, -1].reshape(BACHSIZE,1))

        model.cleargrads()
        loss.backward()
        optimizer.update()

        print "step {} {}".format(k, loss)

        if k%10 == 0:

            print "\n0 = -1, {}".format(model(true_song0).data)
            print "1 =  1, {}\n".format(model(true_song1).data)
