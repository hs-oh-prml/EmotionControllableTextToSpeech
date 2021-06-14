import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os
import argparse
import re
from dataset import Dataset

import hparams as hp
import utils
from loss import SERLoss
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd 
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, step):
    EMOTIONS = {1:'neu', 2:'ang', 3:'dis', 4:'fea', 5:'hap', 6:'sad', 7:'sur'}
    
    model.eval()
    torch.manual_seed(0)

    eval_path = hp.eval_path
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # Get dataset
    dataset = Dataset("val.txt", sort=False)
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=dataset.collate_fn, drop_last=False, num_workers=0, )
    
    # Evaluation
    l = []
    a = []
    pred = []
    gt = []        

    for i, data_of_batch in enumerate(loader):
        # Get Data
        mel = np.array(data_of_batch["mel_target"])
        mel = np.expand_dims(mel, 1)
        mel = torch.from_numpy(mel).float().cuda()

        emotion = torch.from_numpy(np.array(data_of_batch["emotion"])).float().cuda()
        with torch.no_grad():
            # Forward
            v_output_logits, v_output_softmax = model(mel)
            v_predictions = torch.argmax(v_output_softmax,dim=1)
            v_gt = torch.argmax(emotion, dim=1)
            
            v_accuracy = torch.sum(v_gt==v_predictions)/float(len(emotion))
            v_loss = SERLoss(v_output_logits,emotion)
            v_loss = v_loss.item()
            l.append(v_loss)
            a.append(v_accuracy)
            # gt_ = np.zeros(7)
            # gt_[v_gt.cpu().numpy()] = 1            
            # pred_ = np.zeros(7)
            # pred_[v_predictions.cpu().numpy()] = 1
            # gt.append(gt_)
            # pred.append(pred_)

            gt = gt +v_gt.cpu().numpy().tolist()
            pred = pred + v_predictions.cpu().numpy().tolist()

    print("done")

    l = sum(l) / len(l)
    a = sum(a) / len(a)
                    
    str1 = "SER Step {},".format(step)
    str2 = "Loss: {}".format(l)
    str3 = "Accuracy: {}".format(a)

    print("\n" + str1)
    print(str2)
    print(str3)
    # print(len(gt), len(pred))
    cm = confusion_matrix(gt, pred)
    names = ["neu", "ang", "dis", "fea", "hap", "sad", "sur"]
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="Blues") # font size
    file_name = os.path.join(hp.eval_path, 'eval_step_{}.png'.format(step))
    plt.savefig(file_name)
    plt.clf()

    model.train()

    return l, a
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()
    
    # Get model
    model = get_FastSpeech2(args.step).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)
    
    # Init directories
    if not os.path.exists(hp.log_path):
        os.makedirs(hp.log_path)
    if not os.path.exists(hp.eval_path):
        os.makedirs(hp.eval_path)
    evaluate(model, args.step, vocoder)
