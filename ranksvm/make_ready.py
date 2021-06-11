import numpy as np
import scipy
from scipy.sparse import csr_matrix
from rank_svm import *
from scipy.io import savemat

from glob import glob 
import random 
import os
import torch

def run(category):
    emo_label = ["neu", category]
    fea_list = None
    emo_list = []

    MAX_COUNT = 300
    hcount = 0
    ncount = 0

    for emo in emo_label:
        path = "C:/Users/hs_oh/Desktop/data/emotion_kor/{}_3000_16000Hz/features/*.*".format(emo)
        f_list = glob(path)
        print(len(f_list))
        for idx, i in enumerate(f_list):
            f = open(i, "r")
            if idx >= MAX_COUNT:
                break
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

    print(fea_list.shape, len(emo_list))
    print(np.min(fea_list), np.max(fea_list))

    for idx, attr in enumerate(emo_label):
        print("Attribute: {}".format(attr))
        """
        for i, lesser in enumerate(sorted_cat_idx):
            for greater in sorted_cat_idx[i:]:
                print lesser, greater
        """
        S_row = []
        S_column = []
        S_value = []
        S_cnt = 0
        O_row = []
        O_column = []
        O_value = []
        O_cnt = 0
        for i, im1_lab in enumerate(emo_list):
            for j, im2_lab in enumerate(emo_list[i+1:]):
                if im1_lab == im2_lab:
                    S_row.append(S_cnt)
                    S_column.append(i)
                    S_value.append(-1)
                    S_row.append(S_cnt)
                    S_column.append(i + j + 1)
                    S_value.append(1)
                    S_cnt += 1
                    S_row.append(S_cnt)
                    S_column.append(i)
                    S_value.append(1)
                    S_row.append(S_cnt)
                    S_column.append(i + j + 1)
                    S_value.append(-1)
                    S_cnt += 1
                elif im1_lab == "neu" and im2_lab != "neu":
                    O_row.append(O_cnt)
                    O_column.append(i)
                    O_value.append(-1)

                    O_row.append(O_cnt)
                    O_column.append(i + j + 1)
                    O_value.append(1)
                    O_cnt += 1
                elif im1_lab != "neu" and im2_lab == "neu":
                    O_row.append(O_cnt)
                    O_column.append(i)
                    O_value.append(1)

                    O_row.append(O_cnt)
                    O_column.append(i + j + 1)
                    O_value.append(-1)
                    O_cnt += 1


        S = csr_matrix((S_value, (S_row, S_column)),(S_cnt, len(fea_list)))
        O = csr_matrix((O_value, (O_row, O_column)),(O_cnt, len(fea_list)))
        C_O = scipy.matrix(0.1 * np.ones([O_cnt, 1]))
        C_S = scipy.matrix(0.1 * np.ones([S_cnt, 1]))

        X = scipy.matrix(fea_list)

        print("min X: %.4f max X: %.4f" % (np.min(X), np.max(X)))

        mat_path = "./saved_data/mat/dual_{}/{}/".format(MAX_COUNT, attr)
        if not os.path.isdir(mat_path):
            os.makedirs(mat_path)

        savemat(os.path.join(mat_path, "X.mat"),{'X':X})
        savemat(os.path.join(mat_path, "S.mat"),{'S':S})
        savemat(os.path.join(mat_path, "O.mat"),{'O':O})
        savemat(os.path.join(mat_path, "C_O.mat"),{'C_O':C_O})
        savemat(os.path.join(mat_path, "C_S.mat"),{'C_S':C_S})
        # print(type(X))
        w = rank_svm(X, S, O, C_S, C_O)

        weight_path = "./saved_data/"
        if not os.path.isdir(weight_path):
            os.makedirs(weight_path)

        np.save(os.path.join(weight_path, "weights_%s" % (attr)), w)
        #print w
        #print w.shape
        # now train ranksvm for only one attribute, we can extend it later for all attribute
        # break

if __name__=="__main__":
    categories = ["hap", "dis", "ang", "sad", "sur", "fea"]
    for c in categories:
        run(c)