import os
import mlflow


token = os.getenv('MLFLOW_TRACKING_TOKEN')
remote_uri = os.getenv('MLFLOW_TRACKING_URI')

if token:
    mlflow.set_tracking_uri(remote_uri)
    mlflow.set_experiment('ci-build')

else:
    mlflow.set_tracking_uri('http://localhost:5000') 
    mlflow.set_experiment('local-test')
