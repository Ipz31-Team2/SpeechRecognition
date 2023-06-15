from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Завантажити навчену модель
model = load_model('speech_model.h5')

# Список класів
classes = ['yes', 'wow', 'stop', 'right', 'left', 'house', 'happy', 'dog', 'cat', 'bird']

@app.route('/predict', methods=['POST'])
def predict():
    # Отримати аудіозапис з запиту
    audio_file = request.files['audio']
    audio_path = 'audio'
    audio_file.save(audio_path)
    
    # Обробити аудіозапис
    audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    normalized_mfccs = normalize([mfccs])
    
    # Зробити прогноз
    predictions = model.predict(normalized_mfccs)
    predicted_label = classes[np.argmax(predictions)]
    
    # Видалити тимчасовий файл
    os.remove(audio_path)
    
    # Повернути результат у форматі JSON
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
