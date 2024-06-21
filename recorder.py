import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

def record_audio(filename, duration, sample_rate=16000):
    """
    Graba audio desde el micrófono y guarda en un archivo WAV.

    Args:
    filename (str): El nombre del archivo donde se guardará la grabación.
    duration (int): Duración de la grabación en segundos.
    sample_rate (int, opcional): Frecuencia de muestreo. Predeterminado es 16000 Hz.

    Returns:
    None
    """
    print(f"Comenzando la grabación por {duration} segundos...")
    # Grabar audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Esperar hasta que termine la grabación

    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Guardar la grabación en un archivo WAV
    wav.write(filename, sample_rate, recording)
    print(f"Grabación guardada en {filename}")

# Ejemplo de uso: Grabar 5 segundos y guardar en 'grabaciones/test_grabacion.wav'
record_audio('grabaciones/test_grabacion.wav', duration=2)
