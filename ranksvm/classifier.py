import numpy as np
import scipy
from rank_svm import *
import random 
import os
from glob import glob 
import matplotlib.pyplot as plt

def run(category):
    emo_label = ["neu", category]

    MAX_COUNT = 300
    T_MAX = 700
    fea_list = None
    emo_list = []
    file_list = []
    for emo in emo_label:
        path = "C:/Users/hs_oh/Desktop/data/emotion_kor/{}_3000_16000Hz/features/*.*".format(emo)
        f_list = glob(path)
        test_count = 1
        for idx, i in enumerate(f_list):        
            f = open(i, "r")
            if idx <= MAX_COUNT: continue
            if test_count > T_MAX:
                break
            test_count += 1
            file_list.append(os.path.basename(i))
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

    X = np.matrix(fea_list)
    num_attr = len(emo_label)

    attr_weights = []
    for m in range(num_attr):
        w = np.load("./saved_data/weights_{}.npy".format(emo_label[m]))
        attr_weights.append(w.T.tolist()[0])

    attr_weights = np.matrix(attr_weights)
    print(attr_weights.shape)

    neu_score_list = []
    emo_score_list = []

    colors = ["#F2422C", "#E9F587", "#FC9B2D", "#2D44FC", "#261717", "#3EB3FC", "#9D21E6"]

    for idx, e in enumerate(emo_list):
        score = (attr_weights[0] * X[idx].T)[0, 0]
        if e == "neu":
            neu_score_list.append(score)
            # neu_score_list.append(0)
        else:
            emo_score_list.append(score)

    emo_mean = np.mean(emo_score_list)
    neu_mean = np.mean(neu_score_list)

    scaled_emo_score = emo_score_list - emo_mean
    abs_scaled_emo_score = 1 - np.abs(scaled_emo_score)

    scaled_neu_score = neu_score_list - neu_mean
    abs_scaled_neu_score = np.abs(scaled_neu_score)


    plt.hist(
            abs_scaled_neu_score, bins = T_MAX // 10, 
            color = colors[3],
            density=True,
            alpha=0.5,
            label="Abs Scaled neu"
            )
    plt.hist(
            abs_scaled_emo_score, bins = T_MAX // 10, 
            color = colors[4],
            density=True,
            alpha=0.5,
            label="Abs Scaled {}".format(category)
            )    
    plt.legend()
    date = "0524"
    hist_path = "./saved_data/histogram/{}/{}/".format(date, T_MAX)
    if not os.path.isdir(hist_path):
        os.makedirs(hist_path)
    plt.savefig(os.path.join(hist_path, "{}.png".format(category)))
    plt.clf()
    # plt.show()

if __name__=="__main__":
    categories = ["hap", "dis", "ang", "sad", "sur", "fea"]
    for c in categories:
        run(c)
