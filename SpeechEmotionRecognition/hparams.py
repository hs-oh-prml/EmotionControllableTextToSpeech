import os
cleaners = 'korean_cleaners'
### Emotion Korea ###
dataset = "emotion_kor"

data_path = '/hd0/hs_oh/emotion_korea/'
strength_path = "/hd0/hs_oh/emotion_strength"
preprocessed_path = os.path.join("/hd0/hs_oh/", "emotion_korea")

# Checkpoints and synthesis path
checkpoint_path = os.path.join("/hd0/hs_oh/checkpoints/SER/", "cp")
eval_path = os.path.join("/hd0/hs_oh/checkpoints/SER/", "eval")
log_path = os.path.join("/hd0/hs_oh/checkpoints/SER/", "log")
# test_path = os.path.join("/hd0/hs_oh/checkpoints/SER/", "test")
test_path = os.path.join("./", "test")

# Optimizer
batch_size = 128
epochs = 100
acc_steps = 1

total_step = 100000

# Save, log and synthesis
save_step = 1000
eval_step = 500
eval_size = 256
log_step = 10
clear_Time = 20

