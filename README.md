# Emotion Controllable Text-to-Speech based on FastSpeech 2 

![](./assets/model.png)
# Introduction
Recently, speech synthesis research has developed rapidly, and many studies are now underway on Emotional Text-to-Speech (ETTS). However, there are many challenges with ETTS. Not only is it accompanied by considerable financial problems, but it is even more impossible to obtain the emotional strength label data. In this paper, we propose an unsupervised emotion labeling method and a controllable ETTS model to solve these problems. Emotion scores are given to emotion speech data using relative ranking functions, which are converted to emotion vectors and used as conditions in speech synthesis models. The emotion vector represents the emotion category and strength, which makes it easy to control emotional information. Experiments show that the proposed model controlled via emotion vectors can synthesize natural, emotion-expressive speech.

Implementation by PyTorch

Language: Korea

# Train
```
python train.py
```

# Synthesis
```
python synthesis.py --step 500000
```

# Train and synthesis results
## Audio Sample

- Angry: 0.0

<audio controls>
    <source src='./assets/wav/step_100000_acriil_sad_00001772_mel_ang_0.mp3'>
</audio>

- Angry: Weak

<audio controls>
    <source src='./assets/wav/step_100000_acriil_sad_00001772_mel_ang_weak.mp3'>
</audio>

- Angry: Strong

<audio controls>
    <source src='./assets/wav/step_100000_acriil_sad_00001772_mel_ang_strong.mp3'>
</audio>

[Disgust: 0](/assets/wav/step_100000_acriil_sad_00001772_mel_dis_strong.mp3)

## Pretrained model
### FastSpeech2 
[https://drive.google.com/file/d/1_YyQsmE4Dtxl-J5eXJnmcO4O7KKFLmmv/view?usp=sharing](https://drive.google.com/file/d/1_YyQsmE4Dtxl-J5eXJnmcO4O7KKFLmmv/view?usp=sharing)

### Hi-Fi GAN
[https://drive.google.com/file/d/1nqPDjqEr1oq0T7ezuZTfhBKMtHQgmHOL/view?usp=sharing](https://drive.google.com/file/d/1nqPDjqEr1oq0T7ezuZTfhBKMtHQgmHOL/view?usp=sharing)

## Training visualizing
![](./assets/tensorboard.png)
![](./assets/eval.gif)

## Melspectrogram, f0, energy
![](./assets/neu.png)



# References
- Y. Ren, *et al*., "[FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)," *ICLR*, 2021
