import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://mlflow-service:5000")

#mlflow.set_tracking_uri("http://localhost:5000")



X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

with mlflow.start_run():
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="mlops-model"
    )

client = MlflowClient()
latest_version = client.get_latest_versions("mlops-model")[0].version

client.set_model_version_tag(
    name="mlops-model",
    version=latest_version,
    key="stage",
    value="challenger"
)
