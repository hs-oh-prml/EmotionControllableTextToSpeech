import librosa
import numpy as np
from scipy import signal

################
# General Info #
################
sample_rate = 22050
n_fft = 1024
hop_size = 256
win_size = 1024
num_mels = 80
fmin = 55
fmax = 7600
magnitude_power = 1.5
ref_level_db = 20
min_level_db = -100

##############
# Mel option #
##############
signal_normalization = True
allow_clipping_in_normalization = True
symmetric_mels = True
max_abs_value = 5.

################
# Audio option #
################
rescaling = True
rescaling_max = 0.999
trim_silence = False
trim_top_db = 20
trim_fft_size = 512
trim_hop_size = 128
preemphasize = True
mulawquantization = True

def _stft(y):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, pad_mode='constant')

def _linear_to_mel(spectogram):
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert fmax <= sample_rate // 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                           -max_abs_value, max_abs_value)
        else:
            return np.clip(max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value)

    assert S.max() <= 0 and S.min() - min_level_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value
    else:
        return max_abs_value * ((S - min_level_db) / (-min_level_db))

def trim_silence_(wav):
	return librosa.effects.trim(wav, top_db=trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]

def melspectrogram_generation(wav):
    D = _stft(wav)
    L = _linear_to_mel(np.abs(D) ** magnitude_power)
    S = _amp_to_db(L - ref_level_db)
    S = _normalize(S)
    return S

def audio_preprocess(wav):
    if rescaling:  # hparams.rescale = True
        wav = wav / np.abs(wav).max() * rescaling_max
    # M-AILABS extra silence specific
    if trim_silence:  # hparams.trim_silence = True
        wav = trim_silence_(wav)  # Trim leading and trailing silence
    return wav

def preprocess_audio(wav):
    audio = audio_preprocess(wav)
    mel = melspectrogram_generation(audio)
    return audio, mel
