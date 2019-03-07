# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from networks.discriminater.disciminater import Discriminator
from networks.generator.generator import Generator
from individual.agent import BirdAgent
from individual.agent import LogisAgent
import numpy as np
from chainer import optimizers, Variable, serializers
from copy import deepcopy
import chainer.computational_graph as c
import chainer.functions as F
import chainer.links as L
import time
from otherlib.binary_cross_entropy import my_binary_cross_entropy
import matplotlib.pyplot as plt
import os


def run_vs_chaos(EPOCHS=6000, BATCHSIZE=8, SONGSIZE=64):
    birdGAN = BirdAgent(songsize=SONGSIZE, g_units=128, d_units=128)
    birdLogis = LogisAgent(songsize=SONGSIZE, a=1.30)
    param = "130"
    label_GAN = np.zeros((BATCHSIZE, 1)).astype(np.float32)
    label_Logis = np.ones((BATCHSIZE, 1)).astype(np.float32)

    if not os.path.exists("data/a_"+param):
        os.mkdir("data/a_"+param)

    if not os.path.exists("data/a_"+param+"/song"):
        os.mkdir("data/a_"+param+"/song")

    f = open("data/a_"+param+"/loss.csv", "w")

    for k in range(1, EPOCHS+1):
        fsoundGAN = np.random.uniform(-1, 1, size=(BATCHSIZE, 1)).astype(np.float32)
        fsoundLogis = np.random.uniform(-1, 1, size=(BATCHSIZE, 1)).astype(np.float32)

        songGAN = F.concat(birdGAN.gen_song(fsoundGAN, value_chain=False)[1:])
        songLogis = np.concatenate(birdLogis.gen_song(fsoundLogis)[1:], axis=1)

        fake = birdGAN.dis_song(songGAN)
        real = birdGAN.dis_song(songLogis)

        for dis_step in range(1):
            birdGAN.dis.cleargrads()
            loss_dis1 = -F.sum(my_binary_cross_entropy(fake, label_GAN)) / BATCHSIZE
            loss_dis2 = -F.sum(my_binary_cross_entropy(real, label_Logis)) / BATCHSIZE

            loss_dis = loss_dis1 + loss_dis2
            loss_dis.backward()
            birdGAN.opt_dis.update()

        fake = birdGAN.dis_song(songGAN)
        for gen_step in range(1):
            birdGAN.gen.cleargrads()
            loss_gen = F.sum(my_binary_cross_entropy(fake, label_GAN)) / BATCHSIZE

            loss_gen.backward()
            birdGAN.opt_gen.update()

        print k, loss_dis.data, fake.data[0, 0], real.data[0, 0]
        f.write("{} {} {} {} {} {} {}\n".format(k, loss_dis1.data, loss_dis2.data, loss_dis.data, loss_gen.data, fake.data[0, 0], real.data[0, 0]))

        if (k%100 == 0 or k==1):
            filename = "data/a_"+param+"/song/"+str(k)+".png"
            fig = plt.figure(figsize=(16, 8))
            fake_fig = plt.subplot2grid((1, 2), (0, 0))
            real_fig = plt.subplot2grid((1, 2), (0, 1))

            fake_fig.set_ylim(-1.01,1.01)
            real_fig.set_ylim(-1.01,1.01)

            fake_fig.plot(np.arange(SONGSIZE), songGAN[0].data, color="red")
            real_fig.plot(np.arange(SONGSIZE), songLogis[0], color="blue")

            fig.savefig(filename, dpi=50)
            fig.clf()
            plt.close()

    f.close()

    serializers.save_npz("data/a_"+param+"/model_gen.npz", birdGAN.gen)
    serializers.save_npz("data/a_"+param+"/model_dis.npz", birdGAN.dis)
    serializers.save_npz("data/a_"+param+"/opt_gen.npz", birdGAN.opt_gen)
    serializers.save_npz("data/a_"+param+"/opt_dis.npz", birdGAN.opt_dis)

if __name__ == "__main__":
    run_vs_chaos()

