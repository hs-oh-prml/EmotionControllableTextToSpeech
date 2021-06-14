import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
from utils import process_meta, pad_2D, splitIntoChunks

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename, self.text, self.emotion, self.strength = process_meta(os.path.join(hparams.preprocessed_path, filename))
        self.sort = sort

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        name = self.basename[idx]

        # t = self.text[idx]
        e = self.emotion[idx]
        s = self.strength[idx]
        emo_labels = ["neu", "ang", "dis", "fea", "hap", "sad", "sur"]
        emo_embedding = np.zeros(7)
        if e == "neu": 
            emo_embedding[emo_labels.index(e)] = 1
        else:
            emo_embedding[emo_labels.index(e)] = float(s)

        n = np.load(os.path.join(hparams.data_path, e , "{}.npz".format(name)))
        mel_target = n["mel"]

        sample = {
                  "mel_target": mel_target,
                  "emotion": emo_embedding
                  }
        return sample

    def collate_fn(self, batch):

        emotions = [b["emotion"] for b in batch]
        mel_targets = [b["mel_target"].T for b in batch]
        mel_targets = pad_2D(mel_targets)
        # print(mel_targets.shape)
        # mel_targets = np.expand_dims(mel_targets, 1)
        # mel_chunked = []
        # for mel_spec in mel_targets:
        #     chunks = splitIntoChunks(mel_spec, win_size=128,stride=64)
        #     mel_chunked.append(chunks)

        out = {
               "mel_target": mel_targets,
               "emotion": emotions,
            #    "mel_chunk": mel_chunked
               }

        return out

if __name__ == "__main__":
    # Test
    dataset = Dataset('val.txt')
    training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

