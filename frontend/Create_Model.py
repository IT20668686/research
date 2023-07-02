import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Define max_length
max_length = 100

#  Data Preprocessing
dataset_path = "Student_DataSet"
class_folders = os.listdir(dataset_path)
classes = [folder for folder in class_folders if os.path.isdir(os.path.join(dataset_path, folder))]

audio_data = []
labels = []

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(class_path, filename)
            waveform, sample_rate = librosa.load(file_path, sr=None)
            features = librosa.feature.mfcc(waveform, sample_rate)

            # Padding or truncating the feature array to a fixed length
            if features.shape[1] < max_length:
                pad_width = max_length - features.shape[1]
                features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                features = features[:, :max_length]

            audio_data.append(features)
            labels.append(class_name)

# Reshape
audio_data = np.array(audio_data)
audio_data = audio_data.reshape(len(audio_data), -1)

labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size=0.2, random_state=42)

#  Model Training
model = SVC()   
model.fit(X_train, y_train)

# Model Saving
model_filename = "Student_audio_model.pkl"
joblib.dump(model, model_filename)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)
