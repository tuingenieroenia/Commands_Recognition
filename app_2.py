import tkinter as tk
import random
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import joblib
import keras

from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from mfcc_utils_nn import mfcc

# Lista de comandos e imágenes correspondientes
commands = ['bird', 'dog', 'cat', 'house', 'tree']
image_directory = './Imagenes'
audio_directory = './Grabaciones'
model_directory = './models'

# Cargar el modelo CNN entrenado
cnn_model = keras.models.load_model(os.path.join(model_directory, 'cnn_speech_recognition_model.h5'))

# Cargar el escalador y el codificador
scaler = joblib.load(os.path.join(model_directory, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(model_directory, 'label_encoder.pkl'))

# Función para grabar audio
def record_audio(filename, duration, sample_rate=16000):
    print(f"Comenzando la grabación por {duration} segundos...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Esperar hasta que termine la grabación
    wav.write(filename, sample_rate, recording)
    print(f"Grabación guardada en {filename}")

# Función para actualizar la imagen mostrada
def show_new_image():
    global current_command
    current_command = random.choice(commands)
    img_path = os.path.join(image_directory, f"{current_command}.jpg")
    img = Image.open(img_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Función para predecir el comando basado en la red neuronal
def predict_command_nn(file_path, model, scaler, label_encoder, n_mfcc, max_len):
    sample_rate, signal = wav.read(file_path)
    features = mfcc(signal, sample_rate, numcep=n_mfcc)
    
    # Normalizar características
    if len(features) > max_len:
        features = features[:max_len]
    elif len(features) < max_len:
        pad_width = max_len - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    
    features = scaler.transform(features.flatten().reshape(1, -1)).reshape((1, max_len, n_mfcc, 1))
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    predicted_command = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_command

# Función para manejar la grabación y predicción
def handle_record_and_predict():
    audio_path = os.path.join(audio_directory, 'test_grabacion.wav')
    record_audio(audio_path, duration=3)
    predicted_command = predict_command_nn(audio_path, cnn_model, scaler, label_encoder, n_mfcc=13, max_len=55)
    if predicted_command == current_command:
        result = f"¡Correcto! La imagen es {current_command} y dijiste {predicted_command}."
    else:
        result = f"Incorrecto. La imagen es {current_command} pero dijiste {predicted_command}."
    messagebox.showinfo("Resultado", result)

# Configuración de la interfaz gráfica
app = tk.Tk()
app.title("Reconocimiento de Voz con Imágenes")

main_frame = ttk.Frame(app, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

image_label = ttk.Label(main_frame)
image_label.grid(row=0, column=0, padx=10, pady=10)

# Colores personalizados
button_bg_color = '#445634'  # Color de fondo de los botones
button_fg_color = 'black'    # Color del texto de los botones

# Personalización de los botones
record_button = tk.Button(main_frame, text="Grabar y Reconocer", command=handle_record_and_predict,
                          bg=button_bg_color, fg=button_fg_color)
record_button.grid(row=1, column=0, padx=10, pady=10)

new_image_button = tk.Button(main_frame, text="Nueva Imagen", command=show_new_image,
                             bg=button_bg_color, fg=button_fg_color)
new_image_button.grid(row=2, column=0, padx=10, pady=10)

# Mostrar la primera imagen
show_new_image()

app.mainloop()
