# --------------------------------------------------------
# get audio feature (MFCC)
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

#In this document,there are two kinds of method of extracting
#MFCC:1. By adopting librosa 2. By adopting Pytorch

import librosa
from librosa import display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
print(librosa.__version__) #0.6.0
print(np.__version__) #1.13.1
print(matplotlib.__version__)#3.1.0

def fileloader(filepath):
    '''
    Load the Video file and print the sr,duration and length of audio sequence
    '''
    y,sr = librosa.load(filepath)
    drtime = round(librosa.get_duration(y,sr=sr)/60,3)
    y_size = y.size

    print("Sample Rate = {sr}\n".format(sr = sr))
    print("Duration = {drtime} min\n".format(drtime = drtime))
    print("Size of audio seqence = {y_size}\n".format(y_size = y_size))
    return y,sr

def log_mel(y,sr,n_fft=2048,hop_length=512,n_mels=80):
    '''
    Extract the log mel features by utilizing this function
    '''
    mel_spec = librosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.amplitude_to_db(mel_spec)
    return log_mel

def pre_emph(y):
    return pre_emph_y = y[1:]-0.97*y[:-1]

def mfcc(y,sr,n_mfcc=13):
    return mfcc = librosa.feature.mfcc(y=pre_emph, sr=sr, n_mfcc=n_mfcc)






