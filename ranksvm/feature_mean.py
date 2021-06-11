import numpy as np
import scipy
from scipy.sparse import csr_matrix
from rank_svm import *
from rank_svm_pytorch import *
from scipy.io import savemat

from glob import glob 
import random 
import os
import torch

def run():
    emo_label = ["neu", "hap", "dis", "ang", "sad", "sur", "fea"]
    # fea_list = None
    emo_list = []

    MAX_COUNT = 300
    hcount = 0
    ncount = 0

    for emo in emo_label:
        path = "C:/Users/hs_oh/Desktop/data/emotion_kor/{}_3000_16000Hz/features/*.*".format(emo)
        f_list = glob(path)
        # print(len(f_list))
        fea_list = None

        for idx, i in enumerate(f_list):
            f = open(i, "r")
            # if idx >= MAX_COUNT:
            #     break
            while True:
                line = f.readline()
                if not line: break
                line = line.rstrip()
                feature = np.array(line.split(" "), dtype=float)
                emo_list.append(emo)
                if fea_list is None:
                    fea_list = feature
                else:
                    fea_list = np.vstack((fea_list, feature))
            f.close()
        mean = np.mean(fea_list, axis=0)
        std = np.std(fea_list, axis=0)
        np.save("./saved_data/{}_mean.npy".format(emo), mean)
        np.save("./saved_data/{}_std.npy".format(emo), std)

        print("{} Mean".format(emo))
        print(fea_list.shape)
        print(mean.shape)
        print(std.shape)

    print(fea_list.shape, len(emo_list))
    print(np.min(fea_list), np.max(fea_list))

if __name__=="__main__":
    run()