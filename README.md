# Commands Recognition

Este proyecto es un sistema de reconocimiento de comandos utilizando modelos de aprendizaje profundo y extracción de características de audio, como MFCC (coeficientes cepstrales en las frecuencias de Mel). El sistema está diseñado para reconocer comandos de voz y proporcionar una respuesta adecuada.

## Estructura del Proyecto

El repositorio contiene los siguientes archivos y directorios importantes:

- `app.py`: Script principal para ejecutar el reconocimiento de comandos.
- `app_2.py`: Otra versión del script principal con modificaciones.
- `cnn_speech_recognition_model.h5`: Modelo de reconocimiento de voz entrenado utilizando una red neuronal convolucional.
- `decoder.py`: Contiene lógica para decodificar las señales de audio en comandos.
- `enfoque_nn.py`: Implementación de enfoques de redes neuronales para el reconocimiento de comandos.
- `mfcc_extractor.py`: Herramienta para extraer características de audio utilizando MFCC.
- `mfcc_utils.py`: Utilidades para trabajar con MFCC.
- `mfcc_utils_nn.py`: Utilidades para el procesamiento de MFCC enfocadas en redes neuronales.
- `recorder.py`: Script para grabar y preprocesar los comandos de voz.
- `scaler.pkl` y `label_encoder.pkl`: Archivos que contienen el escalador y el codificador de etiquetas para procesar los datos.
- Carpeta `models` y `models_2`: Directorios para almacenar modelos de reconocimiento de voz.
- Carpeta `Imagenes`: Contiene imágenes relacionadas con el proyecto.

## Requisitos

Para ejecutar el proyecto, necesitas instalar los siguientes paquetes:

```bash
pip install -r requirements.txt
