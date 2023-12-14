import librosa
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.concatenate((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.concatenate((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.concatenate((result, mel))
    return result

# Emotions in the dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    file_counter = 0  # Counter for tracking the number of processed files
    for file in glob.glob("C:\\Users\\Aalin\\OneDrive\\Desktop\\major project p1\\Audio_Speech_Actors_01-24\\Actor_*\\*.wav"):
        file_counter += 1
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
        print(f"Processed {file_counter} files...", end="\r")  # Print the progress on the same line
    print()  # Move to the next line after processing all files

    # Convert the lists to numpy arrays
    x, y = np.array(x), np.array(y)

    # Shuffle the data
    x, y = shuffle(x, y, random_state=42)

    return train_test_split(x, y, test_size=test_size, random_state=9)

# Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Initialize the Multi-Layer Perceptron Classifier with five hidden layers
model = MLPClassifier(
    hidden_layer_sizes=(512, 384, 256, 192, 128),  # Five hidden layers
    alpha=0.0001,  # Regularization strength
    max_iter=1500,  # Increase the number of iterations
    learning_rate_init=0.001,  # Initial learning rate
    random_state=42,
    learning_rate='adaptive',  # Adaptive learning rate
    early_stopping=True,  # Early stopping based on validation performance
    validation_fraction=0.2  # Fraction of training data for validation
)

# Create the scaler (use StandardScaler for neural networks)
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model
model.fit(x_train_scaled, y_train)

# Predict for the test set
y_pred = model.predict(x_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
