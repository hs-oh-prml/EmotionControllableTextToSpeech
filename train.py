import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hparams
import hparams as hp
import os
import numpy as np
import argparse
import time
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils
from tqdm import tqdm

def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    
    # Get dataset
    dataset = Dataset("train.txt") 
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True,
        collate_fn=dataset.collate_fn, drop_last=True, num_workers=4)

    # Define model
    model = nn.DataParallel(FastSpeech2()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device) 
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    
    # Training
    model = model.train()

    outer_bar = tqdm(total=hparams.total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    step = args.restore_step
    epoch = 1
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        # Get Training Loader
        if step - 1 >= hparams.total_step: break
        for i, batch in enumerate(loader):
            start_time = time.perf_counter()

            # Get Data
            text = torch.from_numpy(batch["text"]).long().cuda()
            mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
            D = torch.from_numpy(batch["D"]).long().cuda()
            f0 = torch.from_numpy(batch["f0"]).float().cuda()
            energy = torch.from_numpy(batch["energy"]).float().cuda()
            src_len = torch.from_numpy(batch["src_len"]).long().cuda()
            mel_len = torch.from_numpy(batch["mel_len"]).long().cuda()
            max_src_len = np.int(np.max(batch["src_len"], ))
            max_mel_len = np.int(np.max(batch["mel_len"]))
            emotion = torch.from_numpy(np.array(batch["emotion"])).float().cuda()

            # Forward
            mel_output, mel_postnet_output, duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                text, src_len, emotion, mel_len, D, f0, energy, max_src_len, max_mel_len)

            # Cal Loss
            mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                duration_output, D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)

            total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

            # Logger
            t_l = total_loss.item()
            m_l = mel_loss.item()
            m_p_l = mel_postnet_loss.item()
            d_l = d_loss.item()
            f_l = f_loss.item()
            e_l = e_loss.item()

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

            # Update weights
            scheduled_optim.step_and_update_lr()
            scheduled_optim.zero_grad()

            # Print
            if step % hp.log_step == 0:
                Now = time.perf_counter()

                str1 = "Step [{}/{}]:\n".format(
                    step, hparams.total_step)
                str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};\n".format(
                    t_l, m_l, m_p_l, d_l, f_l, e_l)
                str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.\n".format(
                    (Now-Start), (hparams.total_step-step)*np.mean(Time))
                outer_bar.write(str1 + str2 + str3)

            train_logger.add_scalar('Loss/total_loss', t_l, step)
            train_logger.add_scalar('Loss/mel_loss', m_l, step)
            train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, step)
            train_logger.add_scalar('Loss/duration_loss', d_l, step)
            train_logger.add_scalar('Loss/F0_loss', f_l, step)
            train_logger.add_scalar('Loss/energy_loss', e_l, step)

            if step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(step)))
                print("save model at step {} ...".format(step))
                mel_plots = utils.plot_image(mel_target, mel_output, D)
                train_logger.add_figure('Output Mel', mel_plots, global_step=step)

            if step % hp.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    d_l, f_l, e_l, m_l, m_p_l = evaluate(model, step)
                    t_l = d_l + f_l + e_l + m_l + m_p_l
                    vstr1 = "FastSpeech2 Step {},\n".format(step)
                    vstr2 = "Duration Loss: {}\n".format(d_l)
                    vstr3 = "F0 Loss: {}\n".format(f_l)
                    vstr4 = "Energy Loss: {}\n".format(e_l)
                    vstr5 = "Mel Loss: {}\n".format(m_l)
                    vstr6 = "Mel Postnet Loss: {}\n".format(m_p_l)

                    outer_bar.write(vstr1 + vstr2 + vstr3 + vstr4 + vstr5 + vstr6)

                    val_logger.add_scalar('Loss/total_loss', t_l, step)
                    val_logger.add_scalar('Loss/mel_loss', m_l, step)
                    val_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, step)
                    val_logger.add_scalar('Loss/duration_loss', d_l, step)
                    val_logger.add_scalar('Loss/F0_loss', f_l, step)
                    val_logger.add_scalar('Loss/energy_loss', e_l, step)

                model.train()

            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            step += 1
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)
            outer_bar.update(1)
        inner_bar.update(1)
        epoch += 1
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
