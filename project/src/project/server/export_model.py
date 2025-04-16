import bentoml
import yaml
import joblib
from pathlib import Path

config_path = (Path(__file__).resolve().parents[2]) / 'config/params.yaml'

with open(config_path, 'r') as f:
    params = yaml.safe_load(f)

model_name = 'MedicalRegressor'
bentoml_model_name = params.get('bento', {}).get('model_name', 'medical_regressor')
bentoml_model_tag = params.get('bento', {}).get('model_tag', 'latest')

model_output = (Path(__file__).resolve().parents[3]) / 'models/medical_regressor.pkl'

bento_model = bentoml.sklearn.save_model(
    name=f'{bentoml_model_name}:{bentoml_model_tag}',
    model=joblib.load(model_output),
    signatures={
        'predict': {'batchable': True, 'batch_dim': 0}
    },
    metadata={
        'description': 'Random Forest Regressor for medical cost prediction',
        'source': 'project/models',
        'model_name': model_name
    }
)
