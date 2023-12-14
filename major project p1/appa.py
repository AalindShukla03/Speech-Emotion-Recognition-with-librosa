from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from sr5 import extract_feature, scaler, model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', uploaded_file=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('static', 'uploads', filename)
        file.save(file_path)

        emotion = predict_emotion(file_path)
        os.remove(file_path)

        return render_template('index.html', result=f'Predicted Emotion: {emotion}', uploaded_file=filename)

def predict_emotion(file_path):
    feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    feature_scaled = scaler.transform([feature])  # Scale the feature using the trained scaler
    emotion = model.predict(feature_scaled)[0]
    return emotion

if __name__ == '__main__':
    app.run(debug=True)
