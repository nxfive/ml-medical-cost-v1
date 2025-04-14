import yaml
import pandas as pd
from sklearn.model_selection import KFold
from models.optuna_tuning import optimize_model
from models.mlflow_logging import log_best_model
import data.load_data


def main():
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

    cv = KFold(
        n_splits=params['cv']['n_splits'], 
        shuffle=params['cv']['shuffle'],
        random_state=42)
    
    study, best_model, best_params = optimize_model(
        X_train, y_train, params['optuna'], params['model']['features'], cv
        )
    log_best_model(best_model, best_params, study, X_train, X_test, y_train, y_test)
    
    
if __name__ == '__main__':
    main()
