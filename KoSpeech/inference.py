# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor
from glob import glob 
import random 
import os 
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


parser = argparse.ArgumentParser(description='KoSpeech')
parser.add_argument('--model_path', type=str, required=False, default='./outputs/2021-06-04/02-06-47/model.pt')
# parser.add_argument('--audio_path', type=str, required=True)
# parser.add_argument('--audio_dir', type=str, required=True)
parser.add_argument('--device', type=str, required=False, default='cuda')
opt = parser.parse_args()

# GT
# emotion_label = ["ang", "dis", "fea", "hap", "neu", "sad", "sur"]
# f = open(os.path.join("./result", "gt_result.txt"), "w")

# for emo in tqdm(emotion_label):
#     count = 0
#     audio_list = glob(os.path.join("/hd0/cb_im/emotion_kor/{}_3000_16000Hz/wav/".format(emo), "*.wav"))
#     print(len(audio_list))
#     random.shuffle(audio_list)
#     for audio_path in tqdm(audio_list):
#         if count > 1000: break
#         count += 1

#         feature = parse_audio(audio_path, del_silence=True).cuda()
#         input_length = torch.LongTensor([len(feature)]).cuda()
#         vocab = KsponSpeechVocabulary('data/vocab/aihub_character_vocabs.csv')

#         model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
#         if isinstance(model, nn.DataParallel):
#             model = model.module
#         model.eval()

#         if isinstance(model, ListenAttendSpell):
#             model.encoder.device = opt.device
#             model.decoder.device = opt.device
#             # y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
#             y_hats = model.recognize(feature.unsqueeze(0), input_length)
#         elif isinstance(model, DeepSpeech2):
#             model.device = opt.device
#             # y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
#             y_hats = model.recognize(feature.unsqueeze(0), input_length)

#         elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
#             # y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
#             y_hats = model.recognize(feature.unsqueeze(0), input_length)

#         sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
#         sentence = " ".join(sentence[0].split())

#         basename = os.path.basename(audio_path)
#         print(basename)
#         line = "{}|{}\n".format(basename, sentence)
#         print(sentence)
#         f.write(line)
# f.close()

# Hypo
# FS
# audio_list = glob(os.path.join("/hd0/hs_oh/result/fastspeech", "*.wav"))
# f = open(os.path.join("./result", "fastspeech_result.txt"), "w")
# GST
# audio_list = glob(os.path.join("/hd0/hs_oh/result/gst", "*.wav"))
# f = open(os.path.join("./result", "gst_result.txt"), "w")

# GT
# fwav = open("/hd0/hs_oh/emotion_korea/test.txt", 'r')
# audio_list = []
# while True:
#     line = fwav.readline()
#     if not line: break
#     items = line.split("|")
#     audio_list.append(os.path.join("/hd0/cb_im/emotion_kor/{}_3000_16000Hz/wav/".format(items[2]), "{}.wav".format(items[0])))
# #     audio_list = glob(os.path.join("/hd0/cb_im/emotion_kor/{}_3000_16000Hz/wav/".format(emo), "*.wav"))
# f = open(os.path.join("./result", "gt_result.txt"), "w")

# for idx, audio_path in enumerate(audio_list):

#     feature = parse_audio(audio_path, del_silence=True).cuda()
#     input_length = torch.LongTensor([len(feature)]).cuda()
#     vocab = KsponSpeechVocabulary('data/vocab/aihub_character_vocabs.csv')

#     model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
#     if isinstance(model, nn.DataParallel):
#         model = model.module
#     model.eval()

#     if isinstance(model, ListenAttendSpell):
#         model.encoder.device = opt.device
#         model.decoder.device = opt.device

#         # y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
#         y_hats = model.recognize(feature.unsqueeze(0), input_length)
#     elif isinstance(model, DeepSpeech2):
#         model.device = opt.device
#         # y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
#         y_hats = model.recognize(feature.unsqueeze(0), input_length)

#     elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
#         # y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
#         y_hats = model.recognize(feature.unsqueeze(0), input_length)

#     sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
#     sentence = " ".join(sentence[0].split())

#     basename = os.path.basename(audio_path)
#     line = "{}|{}\n".format(basename, sentence)
#     print(sentence)
#     f.write(line)
# f.close()


