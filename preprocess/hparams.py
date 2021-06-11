################################
# Audio Parameters             #
################################
sampling_rate=22050
filter_length=1024
hop_length=256
win_length=1024
n_mel_channels=80
mel_fmin = 0
mel_fmax = 8000

f0_min = 71.0
f0_max = 792.8
energy_min = 0.0
energy_max = 283.72

rescaling_max=0.999

cleaners = 'korean_cleaners'

# Tacotron
signal_normalization = True
allow_clipping_in_normalization = True
symmetric_mels = True
max_abs_value = 5.
min_level_db = -100
preemphasize = False
preemphasis = False
power = 1.5
ref_level_db = 20