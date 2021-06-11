import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import hparams
from utils import pad_1D, pad_2D, process_meta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename, self.text, self.emotion, self.strength = process_meta(os.path.join(hparams.preprocessed_path, filename))
        self.sort = sort

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        name = self.basename[idx]

        # t = self.text[idx]    # Raw text
        e = self.emotion[idx]
        s = self.strength[idx]
        emo_labels = ["ang", "dis", "fea", "hap", "sad", "sur"]
        emo_embedding = np.zeros(6)
        if e != "neu":
            s = float(s) * 2
            s = np.clip(s, 0.0, 1.0)
            emo_embedding[emo_labels.index(e)] = float(s)          # Ex) [0, 0, 0, 0.9, 0, 0]

        n = np.load(os.path.join(hparams.data_path, e, "{}.npz".format(name)))
        t = n["tokens"]         # Character token
        mel_target = n["mel"]
        D = n["duration"]
        f0 = n["scaled_f0"]
        energy = n["scaled_energy"]

        sample = {"id": name,
                  "text": t,
                  "mel_target": mel_target,
                  "D": D,
                  "f0": f0,
                  "energy": energy,
                  "emotion": emo_embedding,
                  "emotion_label": e,
                  }
        return sample

    def collate_fn(self, batch):
        ids = [b["id"] for b in batch]
        texts = [b["text"] for b in batch]
        Ds = [b["D"] for b in batch]
        mel_targets = [b["mel_target"] for b in batch]
        f0s = [b["f0"] for b in batch]
        energies = [b["energy"] for b in batch]
        emotions = [b["emotion"] for b in batch]
        emotion_labels = [b["emotion_label"] for b in batch]

        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print('the dimension of text and duration should be the same')
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)

        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "D": Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel,
               "emotion": emotions,
               "emotion_label": emotion_labels,
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

