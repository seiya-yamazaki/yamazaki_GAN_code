# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def create_figure(filename, column, savename):
    f = open(filename)
    list_log = np.genfromtxt(f, delimiter=' ')
    f.close()


    x = np.arange(start=0, stop=list_log[:, column].shape[0])
    y = list_log[:, column]

    # plt.rcParams["font.size"] = 18
    plt.rcParams["font.size"] = 20

    plt.plot(x, y, linestyle='solid', color="blue", label='loss', linewidth=0.4)

    plt.ylim(0,2.1)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()

    plt.rcParams["font.size"] = 13
    plt.legend()
    plt.savefig(savename)
    plt.clf()
    plt.close()

if __name__ == "__main__":

    for i in xrange(20):
        fname = "../../data/ex1/"+str(i)+"/loss.csv"
        c = 1
        sname = "../../data/ex1/"+str(i)+"_loss_disA.png"
        create_figure(fname, c, sname)


