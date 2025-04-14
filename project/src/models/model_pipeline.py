from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def create_rf_pipeline(features: dict, model: RandomForestRegressor) -> Pipeline:
    """
    Creates a pipeline for a Random Forest regressor with custom preprocessing.
    """
    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), features.get('cat', [])),
            ('rest', 'passthrough', features.get('rest', []))
        ])),
        ('model', model)
    ])
