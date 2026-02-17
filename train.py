from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib

# Cargar dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear modelo
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Entrenar
model.fit(X_train, y_train)

# Predecir
predictions = model.predict(X_test)

# Calcular MAE
mae = mean_absolute_error(y_test, predictions)

print(f"MAE: {mae}")

# MAE es lo que de media se equivoca el modelo en sus predicciones.

# Guardar modelo
joblib.dump(model, "random_forest_model.joblib")