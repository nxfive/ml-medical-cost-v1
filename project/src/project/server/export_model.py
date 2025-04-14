import bentoml
import mlflow
import yaml

with open('../../config/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri('file:../../mlruns')
model_name = 'MedicalRegressor'

sklearn_model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')

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
