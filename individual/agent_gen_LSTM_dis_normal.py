# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from networks.discriminater.disciminater import Discriminator
from networks.generator.generator_LSTM import Generator_LSTM
import numpy as np
from chainer import optimizers, Variable
from copy import deepcopy
import chainer.computational_graph as c
import chainer.functions as F
import chainer.links as L

class BirdAgent():

    def __init__(self,  g_units=32, d_units=32, songsize=64):
        self.gen = Generator_LSTM(n_input=1, n_hidden=g_units, n_output=1)
        self.dis = Discriminator(n_input=songsize, n_hidden=d_units, n_output=1)
        self.opt_gen = optimizers.Adam()
        self.opt_dis = optimizers.SGD()

        self.opt_gen.setup(self.gen)
        self.opt_dis.setup(self.dis)

        self.ssize = songsize

    def gen_song(self, fsound, value_chain=True):

        self.gen.reset_state()

        bsize = fsound.shape[0]
        return_song = [0 for l in range(self.ssize+1)]
        return_song[0] = fsound

        if value_chain:
            for k in range(self.ssize):
                return_song[k + 1] = self.gen(return_song[k].reshape(bsize, 1))
        else:
            for k in range(self.ssize):
                if k==0:
                    return_song[k + 1] = self.gen(return_song[k].reshape(bsize, 1))
                else:
                    return_song[k + 1] = self.gen(return_song[k].data.reshape(bsize, 1))

        return return_song
    
    def dis_song(self, song):
        return self.dis(song)

    def update_gen(self, loss_gen):
        self.gen.cleargrads()
        loss_gen.backward()
        self.opt_gen.update()

    def update_dis(self, loss_dis):
        self.dis.cleargrads()
        loss_dis.backward()
        self.opt_dis.update()

class LogisAgent():

    def __init__(self, a = 1.99, songsize=64):
        self.ssize = songsize
        self.paramA = a

    def gen_song(self, fsound):
        bsize = fsound.shape[0]
        return_song = [0 for l in range(self.ssize+1)]
        return_song[0] = fsound
        
        for k in range(self.ssize):
            return_song[k + 1] = 1 - self.paramA * return_song[k].reshape(bsize, 1) * return_song[k].reshape(bsize, 1)

        return return_song




if __name__ == "__main__":
    agent1 = LogisAgent(songsize=5)
    Bsize = 8
    #fsound = np.random.uniform(-1, 1, size=(Bsize,1)).astype(np.float32)
    fsound = np.ones((Bsize,1)).astype(np.float32) * 0.7

    list1 = agent1.gen_song(fsound)
    
    print list1

    #list2 = F.concat(list1[1:])

    #list3 = agent1.dis_song(list2)
    #g = c.build_computational_graph(list3)

    
    #with open("graph.dot", "w") as o:
    #    o.write(g.dump())


