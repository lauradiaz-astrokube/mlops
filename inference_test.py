import joblib
import numpy as np

# Cargar modelo
model = joblib.load("random_forest_model.joblib")

# Eemplo de datos de entrada
input_data = np.array([[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]])

# Predecir
prediction = model.predict(input_data)

print(f"Predicción: {prediction[0]}")