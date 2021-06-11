import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os
import argparse
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
import hparams as hp
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def evaluate(model, step):
    model.eval()
    torch.manual_seed(0)

    eval_path = hp.eval_path
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # Get dataset
    dataset = Dataset("val.txt", sort=False)
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=dataset.collate_fn, drop_last=False, num_workers=0, )
    
    # Get loss function
    Loss = FastSpeech2Loss().to(device)

    # Evaluation
    d_l = []
    f_l = []
    e_l = []
    mel_l = []
    mel_p_l = []
    current_step = 0
    idx = 0
    for i, data_of_batch in enumerate(loader):
        # Get Data
        id_ = data_of_batch["id"]
        text = torch.from_numpy(data_of_batch["text"]).long().to(device)
        mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
        D = torch.from_numpy(data_of_batch["D"]).int().to(device)
        f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
        energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
        src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
        mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
        max_src_len = np.int(np.max(data_of_batch["src_len"]))
        max_mel_len = np.int(np.max(data_of_batch["mel_len"]))
        emotion = torch.from_numpy(np.array(data_of_batch["emotion"])).float().cuda()

        with torch.no_grad():
            # Forward
            mel_output, mel_postnet_output, duration_output, f0_output, energy_output, src_mask, mel_mask, out_mel_len = model(
                    text, src_len, emotion, mel_len, D, f0, energy, max_src_len, max_mel_len)

            # Cal Loss
            mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                    duration_output, D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)

            d_l.append(d_loss.item())
            f_l.append(f_loss.item())
            e_l.append(e_loss.item())
            mel_l.append(mel_loss.item())
            mel_p_l.append(mel_postnet_loss.item())

            if idx == 0:
                # Run vocoding and plotting spectrogram only when the vocoder is defined
                for k in range(1):
                    basename = id_[k]
                    gt_length = mel_len[k]
                    out_length = out_mel_len[k]
                    mel_target_ = mel_target[k, :gt_length]
                    mel_output_ = mel_output[k, :gt_length]
                    mel_postnet = mel_postnet_output[k, :out_length]

                    mel_target_ = mel_target_.transpose(0, 1).detach()
                    mel_output_ = mel_output_.transpose(0, 1).detach()
                    mel_postnet = mel_postnet.transpose(0, 1).detach()

                    np.save(os.path.join(hp.eval_path, 'eval_step_{}_{}_mel_postnet.npy'.format(step, basename)), mel_postnet.cpu().numpy())
                    np.save(os.path.join(hp.eval_path, 'eval_step_{}_{}_mel_gt.npy'.format(step, basename)), mel_target_.cpu().numpy())
                    np.save(os.path.join(hp.eval_path, 'eval_step_{}_{}_mel.npy'.format(step, basename)), mel_output_.cpu().numpy())

                    f0_ = f0[k, :gt_length]
                    f0_output_ = f0_output[k, :out_length]

                    energy_ = energy[k, :gt_length]
                    energy_output_ = energy_output[k, :out_length]

                    gt_duration = D[k, :gt_length].detach().cpu().numpy()
                    utils.plot_data2([(mel_postnet.cpu().numpy(), f0_output_, energy_output_, gt_duration), (mel_target_.cpu().numpy(), f0_, energy_, gt_duration)],
                        ['Synthesized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(hp.eval_path, 'eval_step_{}_{}.png'.format(step, basename)))
                    idx += 1
                print("done")
        current_step += 1

    d_l = sum(d_l) / len(d_l)
    f_l = sum(f_l) / len(f_l)
    e_l = sum(e_l) / len(e_l)

    mel_l = sum(mel_l) / len(mel_l)
    mel_p_l = sum(mel_p_l) / len(mel_p_l) 

    model.train()

    return d_l, f_l, e_l, mel_l, mel_p_l

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
    evaluate(model, args.step)
