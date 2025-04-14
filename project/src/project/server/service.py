import bentoml
import pandas as pd
from pydantic import BaseModel


class MedicalCostFeatures(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str 
    region: str


@bentoml.service(
    name='medical_regressor_service'
)
class MedicalRegressorService:

    def __init__(self):
        self.model = bentoml.models.get('medical_regressor:random_forest').load_model()

    @bentoml.api()
    def predict(self, medical_features: MedicalCostFeatures):
        medical_features_df = pd.DataFrame([medical_features.model_dump()])
        prediction = self.model.predict(medical_features_df)
        
        return {"charges": prediction[0]}

    @bentoml.api()
    def predict_multiple(self, medical_features_list: list[MedicalCostFeatures]):
        medical_features_dicts = [features.model_dump() for features in medical_features_list]
        medical_features_df = pd.DataFrame(medical_features_dicts)
        predictions = self.model.predict(medical_features_df)
        
        return {"charges": predictions.tolist()}


