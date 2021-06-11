import numpy as np
import scipy
from rank_svm import *
import random 
import os
from glob import glob 
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import RobusterScaler
import pandas as pd
import matplotlib.pyplot as plt


def run():
    emo_label = ["neu", "ang", "dis", "fea", "hap", "sad", "sur"]

    MAX_COUNT = 300
    date = "0606"
    save_path = "./strength_{}/".format(date)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    strength_file = open(os.path.join(save_path, "strength.txt"), 'w')

    for i, emo in enumerate(emo_label):
        if emo != "neu":
            w = np.load("./saved_data/weights_{}.npy".format(emo))
            attr_weights= w.T.tolist()[0]
        else :continue
        # attr_weights = np.matrix(attr_weights)
        emo_path = ["neu"]
        # emo_path = []

        emo_path.append(emo)
        f_list = []

        neu_score_list = []
        emo_score_list = []

        for ep in emo_path:
            path = "C:/Users/hs_oh/Desktop/data/emotion_korea/{}_3000_16000Hz/features/*.*".format(ep)
            temp_list = glob(path)
            print("{} Data: {} ".format(ep, len(temp_list)))

            for j, f_name in enumerate(temp_list):
                f = open(f_name, "r")

                while True:
                    line = f.readline()
                    if not line: break
                    line = line.rstrip()
                    feature = np.matrix(line.split(" "), dtype=float)

                    score = (attr_weights * feature.T)[0, 0]
                    if ep == "neu":

                        neu_score_list.append(score)
                        # neu_score_list.append(0)
                    else:
                        # score = (attr_weights * feature.T)[0, 0]
                        emo_score_list.append(score)
                    # score_list.append(score)
                f.close()

        # score_list = []

        scaler = MinMaxScaler()
        # scaler = RobusterScaler()
        # emo_score_list = np.exp(emo_score_list)
        if emo != "neu":
            print(len(emo_score_list), len(neu_score_list))
            total = emo_score_list + neu_score_list
            print(len(total))
            # norm_emo = scaler.fit_transform(np.array(emo_score_list).reshape(-1, 1))
            # norm_neu = scaler.fit_transform(np.array(neu_score_list).reshape(-1, 1))
            total = scaler.fit_transform(np.array(total).reshape(-1, 1))
            norm_emo = total[:len(emo_score_list)]
            norm_neu = total[len(emo_score_list):]


            emo_mean = np.mean(norm_emo)
            neu_mean = np.mean(norm_neu)

            scaled_emo_score = norm_emo - emo_mean
            abs_scaled_emo_score = 1 - np.abs(scaled_emo_score)
            # abs_scaled_emo_score = 1 + np.log(abs_scaled_emo_score)

            scaled_neu_score = norm_neu - neu_mean
            abs_scaled_neu_score = np.abs(scaled_neu_score)
            # abs_scaled_neu_score = np.log(((abs_scaled_neu_score + 1e-6) * 10))
            
            emo_min = np.min(abs_scaled_emo_score)
            emo_max = np.max(abs_scaled_emo_score)

            # neu_min = np.min(abs_scaled_neu_score)
            # neu_max = np.max(abs_scaled_neu_score)
            # print(abs_scaled_neu_score.shape)
            # print("Neu: min: {} max: {} Emo: min: {} max: {}".format(neu_min, neu_max, emo_min, emo_max))
            colors = ["#F2422C", "#E9F587", "#FC9B2D", "#2D44FC", "#261717", "#3EB3FC", "#9D21E6"]
            plt.hist(
                    # abs_scaled_neu_score, bins = 100, 
                    norm_neu, bins = 100, 

                    color = colors[2],
                    density=True,
                    alpha=0.5,
                    label="neu"
                    )
            plt.hist(
                    # abs_scaled_emo_score, bins = 100, 
                    norm_emo, bins = 100, 
                    color = colors[3],
                    density=True,
                    alpha=0.5,
                    label="{}".format(emo)
                    ) 
            plt.legend()
            hist_path = "./saved_data/histogram/{}/".format(MAX_COUNT, date)
            if not os.path.isdir(hist_path):
                os.makedirs(hist_path)
            plt.savefig(os.path.join(hist_path, "{}.png".format(emo)))
            plt.clf()

        for ep in emo_path:
            path = "C:/Users/hs_oh/Desktop/data/emotion_korea/{}_3000_16000Hz/features/*.*".format(ep)
            temp_list = glob(path)
            for j, f_name in enumerate(temp_list):

                if emo == "neu":
                    # strength_line = "{} {} {:.2f}\n".format(os.path.basename(f_name), emo, abs_scaled_neu_score[j][0])
                    strength_line = "{} {} {:.2f}\n".format(os.path.basename(f_name).replace(".txt", ""), emo, 0)
                else:
                    strength_line = "{} {} {:.2f}\n".format(os.path.basename(f_name).replace(".txt", ""), emo, abs_scaled_emo_score[j][0])
                # print(strength_line)
                strength_file.write(strength_line)
    strength_file.close()

if __name__=="__main__":
    run()    
