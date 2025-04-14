import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from models.model_pipeline import create_rf_pipeline


def objective(trial, X_train, y_train, optuna_params, features, cv):
    n_estimators = trial.suggest_int(
        'n_estimators',
        optuna_params['n_estimators']['min'],
        optuna_params['n_estimators']['max'],
        step=optuna_params['n_estimators']['step']
    )
    max_depth = trial.suggest_int(
        'max_depth',
        optuna_params['max_depth']['min'],
        optuna_params['max_depth']['max']
    )
    min_samples_split = trial.suggest_int(
        'min_samples_split',
        optuna_params['min_samples_split']['min'],
        optuna_params['min_samples_split']['max'],
    )
    min_samples_leaf = trial.suggest_int(
        'min_samples_leaf',
        optuna_params['min_samples_leaf']['min'],
        optuna_params['min_samples_leaf']['max']
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    pipeline = create_rf_pipeline(features, model)

    r2_scores = []

    for step, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

        pipeline.fit(X_train_cv, y_train_cv)
        r2_score = pipeline.score(X_val_cv, y_val_cv)
        r2_scores.append(r2_score)
        trial.report(r2_score, step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return np.mean(r2_scores)


def optimize_model(X_train, y_train, optuna_params, features, cv):
    study = optuna.create_study(
        study_name='MedicalRegressor',
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective(trial, X_train, y_train, optuna_params, features, cv), 
                   n_trials=optuna_params['trials'])

    best_params = study.best_params
    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )

    pipeline = create_rf_pipeline(features, best_model)
    pipeline.fit(X_train, y_train)

    return study, pipeline, best_params

