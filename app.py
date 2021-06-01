from flask import Flask, render_template, request
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc , logfbank
import librosa as lr
import os, glob, pickle
import librosa
from scipy import signal
import noisereduce as nr
from glob import glob
import librosa
import glob


import soundfile
from sklearn.model_selection import train_test_split

model = pickle.load(open('Emotion_Voice_Detection_Model4.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def home():
    file=r'C:\Users\User\OneDrive\Desktop\emotion recognition project\output.wav'
    ans =[]
    new_feature  = extract_feature(file, mfcc=True, chroma=True, mel=True)
    ans.append(new_feature)
    ans = np.array(ans)
    pred=model.predict(ans)
    return render_template('predict.html', data=pred)

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

    #return render_template('prediction.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
