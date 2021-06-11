import numpy as np

import hparams
import hparams as hp
import os
import random
from glob import glob
import utils
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import shutil
#
# emo_labels = ['neu', 'ang', 'dis', 'fea', 'hap', 'sad', 'sur']
# count = 0
# for emo in tqdm(emo_labels):
#     f0_list = []
#     energy_list = []
#     mel_list = []
#     f_list = glob("/home/prml/hs_oh/dataset/emotion_korea/{}/*.*".format(emo))
#     for f in tqdm(f_list):
#         if count == 1: break
#         count = count + 1
#         n = np.load(f)
#         print(n.files)
#         f0 = n["f0"]
#         energy = n["energy"]
#         mel = n['mel']
#         tokens = n["tokens"]
#         duration = n['duration']
#
#         f_name = os.path.basename(f)
#         temp_path = os.path.join("/home/prml/jihyun/dataset/duration_all/duration", f_name)
#         if os.path.exists(temp_path):
#             on = np.load(temp_path)
#             o_duration = on["duration"]
#
#             o_sum = o_duration.sum()
#             d_sum = duration.sum()
#             if o_sum == d_sum:
#                 duration = o_duration
#             else:
#                 print("Duration is not match: {}".format(f_name))
#         else:
#             print("There is no duration file")
#         text = n["text"]
#         s_f0 = utils.average_by_duration(f0, duration)
#         s_energy = utils.average_by_duration(energy, duration)
#         data = {
#             'mel': mel,
#             'f0': f0,
#             "scaled_f0": s_f0,
#             'energy': energy,
#             "scaled_energy": s_energy,
#             'duration': duration,
#             'text': text,
#             'tokens': tokens,
#             'loss_coeff': 1
#         }
        # np.savez(f, **data, allow_pickle=False)

# data = "/home/prml/hs_oh/dataset/emotion_korea2/hap/acriil_hap_00000373.npz"
# n = np.load(data)
# mel = n["mel"]
# mel_norm = n["mel_norm"]
# mel = np.reshape(mel, (0, 1))
# mel_norm = np.reshape(mel_norm, (0, 1))
#
# print(mel.shape)
# print(mel_norm.shape)
# np.save(os.path.join("/home/prml/hs_oh/FastSpeech2/temp", 'temp_mel.npy'), mel)
# np.save(os.path.join("/home/prml/hs_oh/FastSpeech2/temp", 'temp_mel_norm.npy'), mel_norm)

# Mean, Std
# scalers = [StandardScaler(copy=False) for _ in range(3)]  # scalers for mel, f0, energy
# mel_scaler, f0_scaler, energy_scaler = scalers
#
# emo_labels = ['neu', 'ang', 'dis', 'fea', 'hap', 'sad', 'sur']
# for emo in emo_labels:
#     f0_list = []
#     energy_list = []
#     mel_list = []
#     f_list = glob("/home/prml/hs_oh/dataset/emotion_korea/{}/*.*".format(emo))
#     for f in f_list:
#         n = np.load(f)
#         # print(n.files)
#         f0 = n["f0"]
#         energy = n["energy"]
#         mel = n['mel']
#         f0_list.append(f0)
#         energy_list.append(energy)
#         mel_list.append(mel)
#
#     mel_scaler.partial_fit(mel)
#     f0_scaler.partial_fit(f0.reshape(-1, 1))
#     energy_scaler.partial_fit(energy.reshape(-1, 1))
#
#     param_list = [np.array([scaler.mean_, scaler.scale_]) for scaler in scalers]
#     param_name_list = ['{}_mel_stat.npy'.format(emo), '{}_f0_stat.npy'.format(emo), '{}_energy_stat.npy'.format(emo)]
#     [np.save(os.path.join("/home/prml/hs_oh/dataset/emotion_korea/", param_name), param_list[idx]) for idx, param_name in enumerate(param_name_list)]

# Split Data
# strength = []
# train_l = []
# val_l = []
# test_l = []
#
# s_file = open(os.path.join(hp.strength_path, "strength_log.txt"), "r")
# while True:
#     line = s_file.readline()
#     if not line: break
#     # strength = np.vstack(strength, line)
#     strength.append(line)
#
# neu_c = 0
# ang_c = 0
# dis_c = 0
# fea_c = 0
# hap_c = 0
# sad_c = 0
# sur_c = 0
# random.shuffle(strength)
# for line in strength:
#     l = line.split(" ")
#     if l[1] == "neu":
#         if neu_c < 100:
#             test_l.append(line)
#         elif neu_c < 200:
#             val_l.append(line)
#         else:
#             train_l.append(line)
#         neu_c += 1
#     elif l[1] == "ang":
#         if ang_c < 100:
#             test_l.append(line)
#         elif ang_c < 200:
#             val_l.append(line)
#         else:
#             train_l.append(line)
#         ang_c += 1
#     elif l[1] == "dis":
#         if dis_c < 100:
#             test_l.append(line)
#         elif dis_c < 200:
#             val_l.append(line)
#         else:
#             train_l.append(line)
#         dis_c += 1
#     elif l[1] == "hap":
#         if hap_c < 100:
#             test_l.append(line)
#         elif hap_c < 200:
#             val_l.append(line)
#         else:
#             train_l.append(line)
#         hap_c += 1
#     elif l[1] == "sad":
#         if sad_c < 100:
#             test_l.append(line)
#         elif sad_c < 200:
#             val_l.append(line)
#         else:
#             train_l.append(line)
#         sad_c += 1
#     elif l[1] == "sur":
#         if sur_c < 100:
#             test_l.append(line)
#         elif sur_c < 200:
#             val_l.append(line)
#         else:
#             train_l.append(line)
#         sur_c += 1
#     elif l[1] == "fea":
#         if fea_c < 100:
#             test_l.append(line)
#         elif fea_c < 200:
#             val_l.append(line)
#         else:
#             train_l.append(line)
#         fea_c += 1
#     else:
#         print("Error: {}".format(l[1]))
# print(len(train_l), len(val_l), len(test_l))
#
# train = os.path.join(hp.preprocessed_path, "train.txt")
# val = os.path.join(hp.preprocessed_path, "val.txt")
# test = os.path.join(hp.preprocessed_path, "test.txt")
# paths = [train, val, test]
#
# succ_count = 0
# fail_count = 0
#
# for idx, path in enumerate(paths):
#     write_file = open(path, "w")
#     if idx == 0:
#         file_list = train_l
#     elif idx == 1:
#         file_list = val_l
#     elif idx == 2:
#         file_list = test_l
#     else:
#         print("Error")
#         break
#     for line in file_list:
#         l = line.split(" ")
#         info_path = hparams.data_path + "/{}".format(l[1])+ "/" + l[0] + ".npz"
#         if not os.path.exists(info_path):
#             print("NO FILE in:{}".format(info_path))
#             fail_count += 1
#             continue
#
#         n = np.load(info_path)
#         text = n["text"]
#         w_line = "{}|{}|{}|{}".format(l[0], text, l[1], l[2])
#         write_file.write(w_line)
#         succ_count += 1
#         n.close()
#     write_file.close()
# print(succ_count, fail_count)

## Move
path = "/home/prml/hs_oh/checkpoints/FastSpeech2/eval/*.png"
target_path = "/home/prml/hs_oh/checkpoints/FastSpeech2/eval_image/"
f_list = glob(path)
for i in f_list:
    shutil.move(i, target_path)
