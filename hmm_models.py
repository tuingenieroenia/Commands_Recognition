import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Cargar funciones de extracción de características
from mfcc_utils import extract_features_from_directory

# Directorio de datos y comandos a modelar
data_directory = './Grabaciones'
commands = ['bird', 'dog', 'cat', 'go', 'house', 'tree', 'stop']

# Parámetros del HMM
n_components = 12  # Número de estados ocultos
covariance_type = 'diag'  # Tipo de covarianza (diagonal)

# Codificador de etiquetas
label_encoder = LabelEncoder()
label_encoder.fit(commands)

# Diccionario para guardar los modelos HMM
models = {}

# Extraer características y etiquetas
features, labels = extract_features_from_directory(data_directory, max_len=45)

# Normalización
scaler = StandardScaler()
normalized_features = scaler.fit_transform([feat.flatten() for feat in features])
normalized_features = normalized_features.reshape(features.shape)

X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.3, random_state=42)

# Entrenar un HMM para cada comando
for command in commands:
    # Filtrar las características y etiquetas para el comando actual
    command_features = [feat for feat, label in zip(normalized_features, labels) if label == command]
    command_features = np.vstack(command_features)  # Convertir a una sola matriz
    
    # Crear y entrenar el modelo HMM
    model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=200)
    model.fit(command_features)
    
    # Guardar el modelo entrenado
    models[command] = model
    joblib.dump(model, f'hmm_{command}.pkl')
    print(f"Model for {command} trained and saved.")

print("All models trained and saved successfully.")

# Cargar los modelos entrenados
models = {command: joblib.load(f'hmm_{command}.pkl') for command in commands}

# Función para predecir el comando basado en características MFCC
def predict_command(features, models):
    scores = {}
    for command, model in models.items():
        scores[command] = model.score(features)
    return max(scores, key=scores.get)

# Probar el modelo con datos de prueba
correct = 0
total = 0

for i in range(len(X_test)):
    predicted_command = predict_command(X_test[i], models)
    actual_command = y_test[i]
    if predicted_command == actual_command:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")
