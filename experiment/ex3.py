# -*- coding: utf-8 -*-
import sys

sys.path.append("../")
from networks.discriminater.disciminater import Discriminator
from networks.generator.generator import Generator
from individual.agent import BirdAgent
import numpy as np
from chainer import optimizers, Variable
from copy import deepcopy
import chainer.computational_graph as c
import chainer.functions as F
import chainer.links as L
import time
from otherlib.binary_cross_entropy import my_binary_cross_entropy
import matplotlib.pyplot as plt
import os


def run(EPOCHS=5000, BATCHSIZE=64, SONGSIZE=128, DISSTEP=1, GENSTEP=1, NUMBERS=0):
    birdA = BirdAgent(songsize=SONGSIZE, g_units=128, d_units=128)
    birdB = BirdAgent(songsize=SONGSIZE, g_units=128, d_units=128)

    dirname = "ex3"

    function = "../../data/"+str(dirname)+"/"+str(NUMBERS)
    if not os.path.exists(function):
        os.mkdir(function)

    f = open(function + "/loss.csv", "w")

    if not os.path.exists(function + "/image"):
        os.mkdir(function + "/image")


    label0 = np.zeros((BATCHSIZE, 1)) #fakeのラベル
    label1 = np.ones((BATCHSIZE, 1)) #realラベル


    # 初期歌(epoch=0)の保存
    fsoundA = np.random.uniform(-1, 1, size=(BATCHSIZE, 1)).astype(np.float32)
    fsoundB = np.random.uniform(-1, 1, size=(BATCHSIZE, 1)).astype(np.float32)

    songA = F.concat(birdA.gen_song(fsoundA)[1:])
    songB = F.concat(birdB.gen_song(fsoundB)[1:])

    filename = function + "/image/0.png"
    fig = plt.figure(figsize=(16, 8))
    songA_fig = plt.subplot2grid((1, 2), (0, 0))
    songB_fig = plt.subplot2grid((1, 2), (0, 1))

    songA_fig.set_ylim(-1.01, 1.01)
    songB_fig.set_ylim(-1.01, 1.01)

    songA_fig.plot(np.arange(SONGSIZE), songA[0].data, color="red")
    songB_fig.plot(np.arange(SONGSIZE), songB[0].data, color="blue")

    fig.savefig(filename, dpi=50)
    fig.clf()
    plt.close()
    # 初期歌の保存

    for k in range(1, EPOCHS + 1):
        fsoundA = np.random.uniform(-1, 1, size=(BATCHSIZE, 1)).astype(np.float32)
        fsoundB = np.random.uniform(-1, 1, size=(BATCHSIZE, 1)).astype(np.float32)

        # AがrealでBがfakeのとき（Bが真似する・Aが真似される）
        # AのDiscriminatorの学習
        for dis_step in range(DISSTEP):
            # 歌作り
            songA = F.concat(birdA.gen_song(fsoundA)[1:])
            songB = F.concat(birdB.gen_song(fsoundB)[1:])

            A_dis_of_songA = birdA.dis_song(F.ceil(songA * 10) * 0.1)
            A_dis_of_songB = birdA.dis_song(F.ceil(songB * 10) * 0.1)

            # AのDiscriminatorの更新
            birdA.dis.cleargrads()
            loss_disA1 = -F.sum(F.log(A_dis_of_songA)) / BATCHSIZE
            loss_disA2 = -F.sum(F.log(1.0 - A_dis_of_songB)) / BATCHSIZE
            loss_disA = loss_disA1 + loss_disA2
            loss_disA.backward()
            birdA.opt_dis.update()

        # BのGeneratorの学習
        for gen_step in range(GENSTEP):
            # 歌作り
            songB = F.concat(birdB.gen_song(fsoundB)[1:])
            A_dis_of_songB = birdA.dis_song(F.ceil(songB * 10) * 0.1)

            # BのGeneratorの真似する更新
            birdB.gen.cleargrads()
            imitate_loss_genB = F.sum(F.log(1.0 - A_dis_of_songB)) / BATCHSIZE
            imitate_loss_genB.backward()
            birdB.opt_gen.update()

        # AのGeneratorの学習
        for gen_step in range(GENSTEP):
            # 歌作り
            songA = F.concat(birdA.gen_song(fsoundA)[1:])
            A_dis_of_songA = birdA.dis_song(F.ceil(songA * 10) * 0.1)

            # AのGeneratorの真似されないようにする更新
            birdA.gen.cleargrads()
            imitated_loss_genA = -F.sum(F.log(A_dis_of_songA)) / BATCHSIZE
            imitated_loss_genA.backward()
            birdA.opt_gen.update()


        # BがrealでAがfakeのとき（Aが真似する・Bが真似される）
        # BのDiscriminatorの学習
        for dis_step in range(DISSTEP):
            # 歌作り
            songA = F.concat(birdA.gen_song(fsoundA)[1:])
            songB = F.concat(birdB.gen_song(fsoundB)[1:])

            B_dis_of_songA = birdB.dis_song(F.ceil(songA * 10) * 0.1)
            B_dis_of_songB = birdB.dis_song(F.ceil(songB * 10) * 0.1)

            # BのDiscriminatorの更新
            birdB.dis.cleargrads()
            loss_disB1 = -F.sum(F.log(B_dis_of_songB)) / BATCHSIZE
            loss_disB2 = -F.sum(F.log(1.0 - B_dis_of_songA)) / BATCHSIZE
            loss_disB = loss_disB1 + loss_disB2
            loss_disB.backward()
            birdB.opt_dis.update()

        # AのGeneratorの学習
        for gen_step in range(GENSTEP):
            # 歌作り
            songA = F.concat(birdA.gen_song(fsoundA)[1:])
            B_dis_of_songA = birdB.dis_song(F.ceil(songA * 10) * 0.1)

            # AのGeneratorの真似する更新
            birdA.gen.cleargrads()
            imitate_loss_genA = F.sum(F.log(1.0 - B_dis_of_songA)) / BATCHSIZE
            imitate_loss_genA.backward()
            birdA.opt_gen.update()

        # BのGeneratorの学習
        for gen_step in range(GENSTEP):
            # 歌作り
            songB = F.concat(birdB.gen_song(fsoundB)[1:])
            B_dis_of_songB = birdB.dis_song(F.ceil(songB * 10) * 0.1)

            # BのGeneratorの真似されないようにする更新
            birdB.gen.cleargrads()
            imitated_loss_genB = -F.sum(F.log(B_dis_of_songB)) / BATCHSIZE
            imitated_loss_genB.backward()
            birdB.opt_gen.update()


        # 途中歌(epoch=k)の保存
        songA = F.concat(birdA.gen_song(fsoundA)[1:])
        songB = F.concat(birdB.gen_song(fsoundB)[1:])

        A_dis_of_songA = birdA.dis_song(F.ceil(songA * 10) * 0.1)
        A_dis_of_songB = birdA.dis_song(F.ceil(songB * 10) * 0.1)

        B_dis_of_songA = birdB.dis_song(F.ceil(songA * 10) * 0.1)
        B_dis_of_songB = birdB.dis_song(F.ceil(songB * 10) * 0.1)



        print k, loss_disA.data, loss_disB.data, A_dis_of_songB[0], A_dis_of_songA[0], B_dis_of_songA[0], B_dis_of_songB[0]

        f.write("{} {} {} {} {} {} {} {} {} {} {} \n".format(k, loss_disA.data, imitate_loss_genB, imitated_loss_genA, loss_disB.data, imitate_loss_genA, imitated_loss_genB,
                                                            A_dis_of_songB[0].data[0], A_dis_of_songA[0].data[0], B_dis_of_songA[0].data[0], B_dis_of_songB[0].data[0]))

        if ((k % 10 == 0) or k == 1):
            filename = function + "/image/" + str(k) + ".png"
            fig = plt.figure(figsize=(16, 8))
            songA_fig = plt.subplot2grid((1, 2), (0, 0))
            songB_fig = plt.subplot2grid((1, 2), (0, 1))

            songA_fig.set_ylim(-1.01, 1.01)
            songB_fig.set_ylim(-1.01, 1.01)

            songA_fig.plot(np.arange(SONGSIZE), songA[0].data, color="red")
            songB_fig.plot(np.arange(SONGSIZE), songB[0].data, color="blue")

            fig.savefig(filename, dpi=50)
            fig.clf()
            plt.close()

    f.close()


if __name__ == "__main__":

    dirname0 = "ex3"

    function0 = "../../data/"+str(dirname0)
    if not os.path.exists(function0):
        os.mkdir(function0)

    for i in xrange(20):
        run(NUMBERS=i)




















