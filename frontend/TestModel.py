import os
import librosa
import numpy as np
import joblib

# Define max_length
max_length = 100  

#  Class Prediction
def predict_class(audio_file_path):
    waveform, sample_rate = librosa.load(audio_file_path, sr=None)
    features = librosa.feature.mfcc(waveform, sample_rate)

    # Padding feature array to a fixed length
    if features.shape[1] < max_length:
        pad_width = max_length - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_length]

    # Reshape the features array
    features = features.reshape(1, -1) 

    # Load the trained model
    model_filename = "Student_audio_model.pkl"
    loaded_model = joblib.load(model_filename)

    # Predict the class
    predicted_class = loaded_model.predict(features)

    return predicted_class[0]

# word input
input_word = input("Please input Word: ")


# define audio
audio_file_path = "me_mp3.mp3"
predicted_class = predict_class(audio_file_path)

if input_word == predicted_class:
    print("Correct Answer")
else:
    print("worng Answer ")
