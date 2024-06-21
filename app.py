import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import random
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import joblib
from mfcc_utils import mfcc, decode_audio_viterbi

# Lista de comandos e imágenes correspondientes
commands = ['bird', 'dog', 'cat', 'house', 'tree']
image_directory = './Imagenes'
audio_directory = './Grabaciones'
model_directory = './models'

# Cargar los modelos HMM entrenados
models = {command: joblib.load(os.path.join(model_directory, f'baumwelch_hmm_{command}.pkl')) for command in commands}

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

# Función para manejar la grabación y predicción
def handle_record_and_predict():
    audio_path = os.path.join(audio_directory, 'test_grabacion.wav')
    record_audio(audio_path, duration=3)
    predicted_command = decode_audio_viterbi(audio_path, models)
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

record_button = ttk.Button(main_frame, text="Grabar y Reconocer", command=handle_record_and_predict)
record_button.grid(row=1, column=0, padx=10, pady=10)

new_image_button = ttk.Button(main_frame, text="Nueva Imagen", command=show_new_image)
new_image_button.grid(row=2, column=0, padx=10, pady=10)

# Mostrar la primera imagen
show_new_image()

app.mainloop()
