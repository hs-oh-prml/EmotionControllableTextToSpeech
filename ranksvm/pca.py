from glob import glob
import os
import numpy as np

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run():
    emo_label = ["neu", "ang", "dis", "fea", "hap", "sad", "sur"]

    MAX_COUNT = 100

    fea_list = None
    emo_list = []
    file_list = []

    for emo in emo_label:

        path = "C:/Users/hs_oh/Desktop/data/emotion_kor/{}_3000_16000Hz/features/*.*".format(emo)
        f_list = glob(path)
        for idx, i in enumerate(f_list):
            f = open(i, "r")
            file_list.append(os.path.basename(i))
            if idx >= MAX_COUNT: break
            while True:
                line = f.readline()
                if not line: break
                line = line.rstrip()
                feature = np.array(line.split(" "), dtype=float)
                # feature = pd.DataFrame(feature)
                emo_list.append(emo)
                if fea_list is None:
                    fea_list = feature
                else:
                    # fea_list = np.vstack((fea_list, feature))
                    fea_list = np.vstack((fea_list, feature))
            f.close()

    print(fea_list.shape, len(emo_list))

    df = pd.DataFrame(fea_list)
    pca = PCA(n_components=3)
    X_proj = pca.fit_transform(df)

    pDf = pd.DataFrame(data = X_proj)
    print(len(pca.explained_variance_ratio_))
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("PC 1", fontsize = 15)
    ax.set_ylabel("PC 2", fontsize = 15)
    ax.set_title("2 component PCA", fontsize = 20)
    colors = ["#F2422C", "#E9F587", "#FC9B2D", "#2D44FC", "#261717", "#3EB3FC", "#9D21E6"]
    mean_colors = ["#DD422C", "#E9DD87", "#FC9BDD", "#0044FC", "#260017", "#3EB300", "#FF21E6"]

    count = 0
    for emo, color in zip(emo_label, colors):
        temp_x = []
        temp_y = []

        for idx in range(MAX_COUNT):
            temp_x.append(pDf.loc[count * MAX_COUNT + idx, 0])
            temp_y.append(pDf.loc[count * MAX_COUNT + idx, 1])

        count += 1
        meanX = np.mean(temp_x)
        meanY = np.mean(temp_y)
        ax.scatter(
            meanX,
            meanY,
            color = mean_colors[count - 1],
            s = 100,
            label="{} mean".format(emo)
        )
        ax.scatter(
            temp_x,
            temp_y,
            color = color,
            s = 1,
            label=emo
        )
        ax.legend()
        ax.grid()
        plt.savefig("./saved_data/pca_2d_{}.png".format(emo))
        plt.clf()
if __name__=="__main__":
    run()    

