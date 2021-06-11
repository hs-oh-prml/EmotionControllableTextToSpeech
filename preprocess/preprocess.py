#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""
import pyworld as pw
import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import yaml
import hparams

from tqdm import tqdm

from datasets.audio_mel_dataset import AudioDataset
from utils import write_hdf5

from text import text_to_sequence, text_to_phoneme
import torch.nn as nn
import torch
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
# import utils
# from parallel_wavegan.bin.audio_mel_preprocess import melspectrogram_generation
_mel_basis = None
from utils2 import plot_data, remove_outlier, average_by_duration
import pysptk

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav
    
    
def melspectrogram(wav, hparams):
    D = librosa.stft(y=wav, n_fft=hparams.fft_size, hop_length=get_hop_size(hparams), win_length=hparams.win_size)
    S = _amp_to_db(_linear_to_mel(np.abs(D)**hparams.power, hparams), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
       return _normalize(S, hparams)
    return S

def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams.filter_length, hop_length=hparams.hop_length, win_length=hparams.win_length)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))  # min_level_db = -100
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * (
                        (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0,
                           hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * (
                    (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis(hparams):
    assert hparams.mel_fmax <= hparams.sampling_rate // 2

    return librosa.filters.mel(hparams.sampling_rate, hparams.filter_length, n_mels=hparams.n_mel_channels,
                               fmin=hparams.mel_fmin, fmax=hparams.mel_fmax)

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        sampling_rate,
        n_mel_channels,
        mel_fmin,
        mel_fmax,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        # original padding
        p = self.n_fft // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)

        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        energy = torch.norm(magnitude, dim=1)

        S = magnitude.squeeze(0).cpu().detach().numpy()
        # LIBROSA
        f0, m = librosa.piptrack(S=S,
                                sr=self.sampling_rate,
                                n_fft=self.n_fft,
                                fmin=0.0,
                                fmax=8000.0,
                                hop_length=self.hop_length,
                                # win_length=self.win_length,
                                # window=self.window
                                )
        pitch = []
        for i in range(0, f0.shape[1]):
            index = m[:, i].argmax()
            pitch.append(f0[index, i])

        # PYSPTK
        # f0 = pysptk.rapt(audio.squeeze(0).cpu().detach().numpy() * 32768.0,
        #                  fs=self.sampling_rate,
        #                  hopsize=self.hop_length,
        #                  min=150.0,
        #                  max=500.0
        #                  )
        #
        # print(f0.shape)

        # print(f0.shape)
        # print(m.shape)
        # print(len(pitch))
        # energy = torch.norm(magnitude, dim=1)
        # f0 = torch.norm(torch.from_numpy(f0), dim=1)

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec, energy, pitch # Tensor, Tensor, List

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio,
                          n_fft=fft_size,
                          hop_length=hop_size,
                          win_length=win_length,
                          window=window,
                          pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py).")
    parser.add_argument("--wav-scp", "--scp", default=None, type=str,
                        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.")
    parser.add_argument("--segments", default=None, type=str,
                        help="kaldi-style segments file. if use, you must to specify both scp and segments.")
    parser.add_argument("--rootdir", default='/cb_im/datasets/emotion_kor/', type=str,
                        help="directory including wav files. you need to specify either scp or rootdir.")
    parser.add_argument("--dumpdir", type=str, required=False, default='/home/prml/hs_oh/dataset/emotion_korea2',
                        help="directory to dump feature files.")
    parser.add_argument("--config", type=str, required=False, default='parallel_wavegan.v1_2.yaml',
                        help="yaml format configuration file.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('Skip DEBUG/INFO messages')

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    assert config['sequence_type'] in ['character', 'phoneme'], 'Wrong sequence Type!'

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or \
            (args.wav_scp is None and args.rootdir is None):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    if args.rootdir is not None:
        dataset = AudioDataset(
            args.rootdir, '/cb_im/datasets/emotion_kor/final.csv',
            "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)
    fft = Audio2Mel(config["fft_size"], config["hop_size"], config["win_length"],config["sampling_rate"],
                    config["num_mels"], config["fmin"], config["fmax"] ).cuda()
    # process each data
    for i, d in enumerate(tqdm(dataset)):
        # check
        # assert len(audio.shape) == 1, \
        #     f"{utt_id} seems to be multi-channel signal."
        # assert np.abs(audio).max() <= 1.0, \
        #     f"{utt_id} seems to be different from 16 bit PCM."
        # # assert fs == config["sampling_rate"], \
        #     f"{utt_id} seems to have a different sampling rate."

        # wav_path, text, spk_id, spk_name = d
        wav_path, text = d

        wav = os.path.join(args.rootdir, wav_path)
        if os.path.exists(wav):
            audio, _ = librosa.load(wav, sr=config['sampling_rate'])
            # print(np.max(audio), np.min(audio))
            # f0 = pysptk.rapt(audio * 32768.0,
            #                  fs=22050,
            #                  hopsize=256,
            #                  min=150.0,
            #                  max=500.0
            #                  )
            # print(f0.shape)

            # scaling
            audio = audio / np.abs(audio).max() * config['max_audio_scale']

        # trimming
            if config["trim_silence"]:
                audio, _ = librosa.effects.trim(audio,
                                            top_db=config["trim_threshold_in_db"],
                                            frame_length=config["trim_frame_size"],
                                            hop_length=config["trim_hop_size"])

            emo = d[0].split("/")[0]
            emo = emo.split("_")[0]
            duration = np.load(os.path.join("/home/prml/sb_kim/output/duration/emotion_kor/{}".format(d[0].replace("_3000_16000Hz/wav", "").replace(".wav", ".npz"))))["duration"]

            mel, energy, f0 = fft(torch.from_numpy(audio)
                              .unsqueeze(0)
                              .unsqueeze(0)
                              .float()
                              .cuda()
                              )
            mel = mel.squeeze(0).cpu().numpy().T

            energy = energy.squeeze(0).cpu().numpy()
            energy = remove_outlier(energy)

            f0 = np.array(f0)
            f0 = remove_outlier(f0)
            # [T, 80]
  
            # Norm Mel
            if hparams.signal_normalization:
                D = _stft(preemphasis(audio, hparams.preemphasis, hparams.preemphasize), hparams)
                S = _amp_to_db(_linear_to_mel(np.abs(D) ** hparams.power, hparams), hparams) - hparams.ref_level_db
                mel_norm = _normalize(S, hparams)    # [80, T]
            
                assert mel.shape[0] == mel_norm.shape[1], 'Mel shape and Normalized Mel shape are unequal!!!' 


            # calculate normalized mel-spectrogram


            # make sure the audio length and feature length are matched
            audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
            audio = audio[:len(mel) * config["hop_size"]]
            # print(len(mel), len(audio))
            assert len(mel) * config["hop_size"] == len(audio)

            # apply global gain
            if config["global_gain_scale"] > 0.0:
                audio *= config["global_gain_scale"]

            # token
            tokens = text_to_sequence(text) if config['sequence_type'] == 'character' else text_to_phoneme(text)

            # save
            data = {
                    # 'audio': audio,
                    'mel': mel,
                    # 'mel_norm': mel_norm.T,
                    'mel_norm': mel_norm.T,
                    'f0': f0,
                    'energy': energy,
                    'duration': duration,
                    'text': text,
                    'tokens': tokens,
                    # 'spk_id': spk_id,
                    # 'spk_name': spk_name,
                    'loss_coeff': 1
                    }
            # print(data.keys())
            wav_path_split = wav_path.split('/')
            # if not os.path.exists(os.path.join(args.rootdir, wav_path_split[0], 'npz')) :
            #     os.mkdir(os.path.join(args.rootdir, wav_path_split[0], 'npz'))

            # print(wav_name.replace('wav', 'npz'))
            # np.savez(os.path.join(root_dir, wav_path_split[0], 'npz', wav_name.replace('.wav', '.npz')), **data, allow_pickle=False)
            # print(wav.replace('wav', 'npz'))
            if i % 1000 == 0:
                fname = wav_path.split("/")[-1]
                fname = fname.replace(".wav", ".png")
                plot_data([(mel.T, f0.T, energy.T)], "Spectrogram", "./mel_f0_energy/{}".format(fname))
                print(i, fname, mel.shape, energy.shape, f0.shape)

            os.makedirs(os.path.join(args.dumpdir, emo), exist_ok=True)
            np.savez(os.path.join(args.dumpdir, emo, os.path.basename(wav_path).replace('wav', 'npz')), **data, allow_pickle=False)

            # utils.plot_data((mel_norm.T, f0, energy),
            #                 ['Synthesized Spectrogram', 'Ground-Truth Spectrogram'],
            #                 filename="./mel_f0_energy/{}.png".format(i))
        """
        if config["format"] == "hdf5":
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "wave", audio.astype(np.float32))
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "feats", mel.astype(np.float32))
        elif config["format"] == "npy":
            np.save(os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                    audio.astype(np.float32), allow_pickle=False)
            np.save(os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                    mel.astype(np.float32), allow_pickle=False)
        else:
            raise ValueError("support only hdf5 or npy format.")
        """
        

if __name__ == "__main__":
    main()
