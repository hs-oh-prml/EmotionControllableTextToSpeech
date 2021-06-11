import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import Dataset
import hparams
import hparams as hp
import os

import argparse
from fastspeech2 import FastSpeech2
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, batch, strength, e_label):
    basename = batch["id"]
    text = torch.from_numpy(batch["text"]).long().cuda()
    D = torch.from_numpy(batch["D"]).long().cuda()
    src_len = torch.from_numpy(batch["src_len"]).long().cuda()
    emo_labels = ["ang", "dis", "fea", "hap", "sad", "sur"]
    emo = np.zeros(6)
    emo[emo_labels.index(e_label)] = strength
    emotion = torch.from_numpy(emo).float().cuda().unsqueeze(0)

    mel, mel_postnet, duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, emotion)
    mel_postnet_torch = mel_postnet.transpose(1, 2)
    f0_output = f0_output[0]
    energy_output = energy_output[0]

    np.save(os.path.join(hp.test_path, 'step_{}_{}_mel_{}x{}.npy'.format(args.step, basename[0], e_label, strength)), mel_postnet_torch[0].cpu().detach().numpy())
    utils.plot_data2([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output, D[0])], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}_{}x{}.png'.format(args.step, basename, e_label, strength)))

    print("{} {}x{} Done".format(basename[0], e_label, strength))


if __name__ == "__main__":
    # Test

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()

    dataset = Dataset('test.txt')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)
    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    model = get_FastSpeech2(args.step).to(device)
    strength = [0.0, 0.3, 1.0]
    emo_labels = ["ang", "dis", "fea", "hap", "sad", "sur"]
    for idx, batch in enumerate(loader):

        for emo in emo_labels:
            for s in strength:
                synthesize(model, batch, s, emo)
