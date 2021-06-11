import os
cleaners = 'korean_cleaners'

audio_data_path = os.path.join("/cb_im/datasets/", dataset)
data_path = '/home/prml/hs_oh/dataset/emotion_korea/'
duration_path = "/home/prml/jihyun/dataset/duration_all/duration"
strength_path = "/home/prml/hs_oh/dataset/emotion_strength"

# Text
text_cleaners = ['korean_cleaners']

# Audio and mel
### Emotion Korea ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0
mel_fmax = 8000

f0_min = 71.0
f0_max = 792.8
energy_min = 0.0
energy_max = 283.72

# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 10000

# Checkpoints and synthesis path
preprocessed_path = os.path.join("/home/prml/hs_oh/dataset/", "emotion_korea")
checkpoint_path = os.path.join("/home/prml/hs_oh/checkpoints/FastSpeech2/", "cp")
eval_path = os.path.join("/home/prml/hs_oh/checkpoints/FastSpeech2/", "eval")
log_path = os.path.join("/home/prml/hs_oh/checkpoints/FastSpeech2/", "log")
test_path = os.path.join("/home/prml/hs_oh/checkpoints/FastSpeech2/", "test")


# Optimizer
batch_size = 48
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.

total_step = 100000

# Save, log and synthesis
save_step = 5000
eval_step = 500
eval_size = 256
log_step = 10
clear_Time = 20

