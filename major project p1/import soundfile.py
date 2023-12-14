import soundfile
import numpy as np
import librosa
import speech_recognition as sr
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature_from_audio(audio_data, mfcc, chroma, mel, sample_rate):
    # Convert byte string to NumPy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Convert integer audio data to floating-point
    X = audio_array.astype(np.float32) / np.iinfo(np.int16).max

    if chroma:
        stft = np.abs(librosa.stft(X))
    result = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result.append(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result.append(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result.append(mel)
    return np.concatenate(result)

# DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    # ... (your existing code for loading data)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# DataFlair - Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# DataFlair - Get the shape of the training and testing datasets
print(f'Total features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi-Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# DataFlair - Train the model
model.fit(x_train, y_train)

# DataFlair - Initialize the recognizer
recognizer = sr.Recognizer()

# DataFlair - Capture audio from the microphone
with sr.Microphone() as source:
    print("Say something:")
    recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
    audio = recognizer.listen(source, timeout=5)

# DataFlair - Use Google Web Speech API to recognize the speech
try:
    text = recognizer.recognize_google(audio)
    print(f"Recognized text: {text}")

    # DataFlair - Extract features from the recognized speech
    features = extract_feature_from_audio(audio.get_raw_data(), mfcc=True, chroma=True, mel=True, sample_rate=audio.sample_rate)

    # DataFlair - Reshape the features to match the training data
    features = features.reshape(1, -1)

    # DataFlair - Predict the emotion
    predicted_emotion = model.predict(features)[0]
    print(f"Predicted emotion: {predicted_emotion}")

except sr.UnknownValueError:
    print("Could not understand audio.")
except sr.RequestError as e:
    print(f"Error making the request: {e}")
except Exception as e:
    print(f"Error processing audio: {e}")
