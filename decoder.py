import numpy as np
import joblib
import os
from scipy.io import wavfile
from mfcc_utils import mfcc

# Lista de comandos
commands = ['bird', 'dog', 'cat', 'go', 'house', 'tree', 'stop']

# Cargar los modelos HMM entrenados
models = {command: joblib.load(f'baumwelch_hmm_{command}.pkl') for command in commands}

# Función para predecir el comando basado en características MFCC usando Viterbi
def predict_command_viterbi(features, models):
    scores = {}
    for command, model in models.items():
        try:
            logprob, state_sequence = model.decode(features, algorithm="viterbi")
            scores[command] = logprob
        except Exception as e:
            print(f"Error decoding {command}: {e}")
            scores[command] = -np.inf
    return max(scores, key=scores.get)

# Función para procesar un archivo de audio y predecir el comando
def decode_audio_viterbi(file_path, models):
    sample_rate, signal = wavfile.read(file_path)
    features = mfcc(signal, sample_rate)

    # Asegurarse de que las características tengan la misma longitud que durante el entrenamiento
    max_len = 50  # Debe coincidir con el valor usado en extract_features_from_directory
    if len(features) > max_len:
        features = features[:max_len]
    elif len(features) < max_len:
        pad_width = max_len - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')

    predicted_command = predict_command_viterbi(features, models)
    return predicted_command

# Ejemplo de uso: Decodificar un archivo de audio
test_file = './Grabaciones/test_sample.wav'  # Reemplaza con el camino a tu archivo de prueba
predicted_command = decode_audio_viterbi(test_file, models)
print(f"Predicted Command using Viterbi: {predicted_command}")