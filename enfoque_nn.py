import numpy as np
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mfcc_utils_nn import extract_features_with_mfcc_length, mfcc

# Configuración de datos
data_directory = './Grabaciones'
commands = ['bird', 'dog', 'cat', 'house', 'tree']
n_mfcc = 13  # Número de coeficientes MFCC a usar
max_len = 55  # Longitud máxima de las características

# Extraer características MFCC
features, labels = extract_features_with_mfcc_length(data_directory, n_mfcc, max_len, commands)

# Aplanar características para la normalización
X_flattened = [feat.flatten() for feat in features]
X_flattened = np.array(X_flattened)

# Normalizar características
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_flattened)

# Codificar etiquetas
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Verifica las formas de los conjuntos de entrenamiento y prueba
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Verifica que el tamaño sea divisible por el nuevo formato
num_samples = X_normalized.shape[0]
total_elements = num_samples * max_len * n_mfcc

if X_normalized.size == total_elements:
    X_train_reshaped = X_train.reshape((X_train.shape[0], max_len, n_mfcc, 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], max_len, n_mfcc, 1))
    print(f"Reshaped train features shape: {X_train_reshaped.shape}")
    print(f"Reshaped test features shape: {X_test_reshaped.shape}")
else:
    raise ValueError(f"Error: expected total elements {total_elements}, but got {X_normalized.size}")

# Guardar el escalador y el codificador para su uso posterior
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(f"Train set: {X_train_reshaped.shape}, Test set: {X_test_reshaped.shape}")

# Convertir etiquetas a formato categórico
num_classes = len(commands)
y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

# Redimensionar datos para el modelo CNN
X_train_reshaped = X_train.reshape((X_train.shape[0], max_len, n_mfcc, 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], max_len, n_mfcc, 1))

# Verifica las formas después del redimensionamiento
print(f"Reshaped train features shape: {X_train_reshaped.shape}")
print(f"Reshaped test features shape: {X_test_reshaped.shape}")

# Definir el modelo CNN
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(max_len, n_mfcc, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenar el modelo
history = model.fit(X_train_reshaped, y_train_categorical, epochs=50, batch_size=32, validation_split=0.2)

# Guardar el modelo entrenado
model.save('cnn_speech_recognition_model.h5')

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_categorical)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Función para predecir el comando basado en audio
def predict_command_nn(file_path, model, scaler, label_encoder, n_mfcc, max_len):
    sample_rate, signal = wavfile.read(file_path)
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

# Ejemplo de uso
test_file = './Grabaciones/test_grabacion.wav'
predicted_command = predict_command_nn(test_file, model, scaler, label_encoder, n_mfcc, max_len)
print(f"Predicted Command using Neural Network: {predicted_command}")

# Gráfica de precisión y pérdida
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()