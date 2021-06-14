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
from loss import SERLoss
from dataset import Dataset
from validation import evaluate
import utils
from model import SERModel

def main(args):
    EMOTIONS = {1:'neu', 2:'ang', 3:'dis', 4:'fea', 5:'hap', 6:'sad', 7:'sur'}
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    
    # Get dataset
    dataset = Dataset("train.txt") 
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True,
        collate_fn=dataset.collate_fn, drop_last=True, num_workers=4)

    # Define model
    model = nn.DataParallel(SERModel(len(EMOTIONS))).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of SER Model Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)
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

    step = args.restore_step
    for epoch in range(hp.epochs):
        # Get Training Loader
        for i, batch in enumerate(loader):
            # if step >= hparams.total_step: break
            start_time = time.perf_counter()

            # Get Data
            mel = np.array(batch["mel_target"])
            mel = np.expand_dims(mel, 1)
            # print(mel.shape)
            mel = torch.from_numpy(mel).float().cuda()

            emotion = torch.from_numpy(np.array(batch["emotion"])).float().cuda()            
            # X: Mel, Y: Label

            # Forward
            output_logits, output_softmax = model(mel)
            predictions = torch.argmax(output_softmax, dim=1)
            gt = torch.argmax(emotion, dim=1)

            accuracy = torch.sum(gt==predictions)/float(len(gt))
            # Compute loss
            # print(emotion[0], output_logits[0])
            loss = SERLoss(output_logits, emotion)
            # Compute gradients
            loss.backward()
            # Update parameters and zero gradients
            optimizer.step()
            optimizer.zero_grad()            # Cal Loss

            # Logger
            l = loss.item()

            # Print
            if step % hp.log_step == 0:
                Now = time.perf_counter()
                str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                    epoch+1, hp.epochs, step, hparams.total_step)
                str2 = "Loss: {:.4f}, ACCURACY: {:.4f}".format(l, accuracy*100)
                str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (hparams.total_step-step)*np.mean(Time))
                print("\n" + str1)
                print(str2)
                print(str3)

            train_logger.add_scalar('loss', l, step)
            train_logger.add_scalar('accuracy', accuracy*100, step)

            if step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(step)))
                print("save model at step {} ...".format(step))

            if step % hp.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    v_loss, v_accuracy = evaluate(model, step)
                    val_logger.add_scalar('val loss', v_loss, step)
                    val_logger.add_scalar('val accuracy', v_accuracy*100, step)
                model.train()

            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            step += 1
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
