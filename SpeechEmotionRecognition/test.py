import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import Dataset
import hparams
import hparams as hp
import os
import argparse

from sklearn.metrics import confusion_matrix
import seaborn as sn
from model import SERModel
from loss import SERLoss
import pandas as pd 
import matplotlib.pyplot as plt
from glob import glob 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    EMOTIONS = {1:'neu', 2:'ang', 3:'dis', 4:'fea', 5:'hap', 6:'sad', 7:'sur'}

    test_path = hp.test_path
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(args.step))
    model = nn.DataParallel(SERModel(len(EMOTIONS)))
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()

    dataset = Dataset("test.txt", sort=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)


    LOAD_PATH = os.path.join(os.getcwd(),'models')

    print('Model is loaded from {}'.format(checkpoint_path))
    t_l = []
    t_a = []
    pred = []
    gt = []    

    f_list = glob("/hd0/hs_oh/result/fastspeech_mel/*")
    for idx, batch in enumerate(f_list):

    # for idx, batch in enumerate(loader):
    #     mel = np.array(batch["mel_target"])
        mel = np.load(batch)
        # print(mel.shape)
        # mel = mel.T # GST 
        # print(mel.shape)

        mel = np.expand_dims(mel, 0)
        mel = np.expand_dims(mel, 1)

        # print(mel.shape)

        mel = torch.from_numpy(mel).float().cuda()
        # strength = os.path.basename(batch).split("_")[-1]

        # strength = os.path.basename(batch).split("_")[-1]
        # strength = strength.replace(".npy", "")        
        # if strength == "gt": continue
        # strength = strength.split("x")[1]
        
        # if strength != "0.5": continue

        e = os.path.basename(batch).split("_")[3]

        # print(e)
        emo_labels = ["neu", "ang", "dis", "fea", "hap", "sad", "sur"]
        emo_embedding = np.zeros(7)
        emo_embedding[emo_labels.index(e)] = 1

 
        emotion = torch.from_numpy(np.array(emo_embedding)).float().cuda()
        emotion = emotion.unsqueeze(0)
        # emotion = torch.from_numpy(np.array(batch["emotion"])).float().cuda()            
 
        with torch.no_grad():
            output_logits, output_softmax = model(mel)
            # print(output_softmax.shape)
            t_predictions = torch.argmax(output_softmax, dim=1)
            # print(emotion.shape)
            t_gt = torch.argmax(emotion, dim=1)

            test_acc = torch.sum(t_gt==t_predictions)/float(len(t_gt))
            test_loss = SERLoss(output_logits, emotion)
            t_l.append(test_loss)
            t_a.append(test_acc)
            gt = gt + t_gt.cpu().numpy().tolist()
            pred = pred + t_predictions.cpu().numpy().tolist()
    l = sum(t_l) / len(t_l)
    a = sum(t_a) / len(t_a)
    a = a * 100
    str1 = "SER Step {},".format(args.step)
    str2 = "Loss: {}".format(l)
    str3 = "Accuracy: {}%".format(a)

    print("\n" + str1)
    print(str2)
    print(str3)
    cm = confusion_matrix(gt, pred)
    names = ["neu", "ang", "dis", "fea", "hap", "sad", "sur"]
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    # plt.figure(figsize=(10,7))
    plt.figure(figsize=(8, 6))
    # sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, cmap="Blues") # font size
    file_name = os.path.join(hp.test_path, 'test_step_{}.png'.format(args.step))
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nAccuracy={:0.4f}'.format(a))
    plt.savefig(file_name)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()

    main(args)
