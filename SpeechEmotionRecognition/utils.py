import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import hparams as hp
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        emotion = []
        strength = []
        for line in f.readlines():
            n, t, e, s = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
            emotion.append(e)
            strength.append(s)
        return name, text, emotion, strength

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[0]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[1]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:s, :]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def splitIntoChunks(mel_spec,win_size,stride):
    t = mel_spec.shape[1]
    num_of_chunks = int(t/stride)
    chunks = []
    for i in range(num_of_chunks):
        chunk = mel_spec[i*stride:i*stride+win_size, :]
        if chunk.shape[1] == win_size:
            chunks.append(chunk.T)
    return np.stack(chunks,axis=0)
