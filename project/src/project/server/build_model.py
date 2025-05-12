import bentoml
import mlflow
import yaml
import joblib
from pathlib import Path
from datetime import datetime


with open('../../config/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri('file:../../mlruns')
model_name = 'MedicalRegressor'

sklearn_model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')

model_tag = datetime.now().strftime("%Y%m%d%H%M%S")  
model_output = (Path(__file__).resolve().parents[3]) / f'models/medical_regressor_{model_tag}.pkl'

joblib.dump(sklearn_model, model_output)

bentoml_model_name = params.get('bento', {}).get('model_name', 'medical_regressor')
bentoml_model_tag = params.get('bento', {}).get('model_tag', 'latest')

bento_model = bentoml.sklearn.save_model(
    name=f'{bentoml_model_name}:{bentoml_model_tag}',
    model=sklearn_model,
    signatures={
        'predict': {'batchable': True, 'batch_dim': 0}
    },
    metadata={
        'description': 'Random Forest Regressor for medical cost prediction',
        'source': 'Mlflow Registry',
        'model_name': model_name
    }
)
