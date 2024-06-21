import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Funciones de procesamiento de audio definidas anteriormente
from mfcc_utils import extract_features_from_directory

# Extracción de características
data_directory = './Grabaciones'
commands = ['bird', 'dog', 'cat', 'house', 'tree']
features, labels = extract_features_from_directory(data_directory)

# Normalización
scaler = StandardScaler()
normalized_features = scaler.fit_transform([feat.flatten() for feat in features])
normalized_features = normalized_features.reshape(features.shape)

X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.2, random_state=42)

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
