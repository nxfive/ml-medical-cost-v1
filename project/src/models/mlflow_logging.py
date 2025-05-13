from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import mlflow
import os 


def setup_mlflow():
    token = os.getenv('MLFLOW_TRACKING_TOKEN')
    remote_uri = os.getenv('MLFLOW_TRACKING_URI')

    if token:
        mlflow.set_tracking_uri(remote_uri)
        mlflow.set_experiment('ci-build')

    else:
        mlflow.set_tracking_uri('http://localhost:5000') 
        mlflow.set_experiment('local-test')


def log_best_model(best_model, best_params, study, X_train, X_test, y_train, y_test):
    
    setup_mlflow()

    with mlflow.start_run(run_name='Best Model', nested=True):
        for param, value in best_params.items():
            mlflow.log_param(f'Best_{param}', value)

        mlflow.log_metric('Best_score', study.best_value)

        y_pred = best_model.predict(X_test)
        test_score = best_model.score(X_test, y_test)
        train_score = best_model.score(X_train, y_train)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        mlflow.log_metric('test_score', test_score)
        mlflow.log_metric('train_score', train_score)
        mlflow.log_metric('mean_absolute_error', mae)
        mlflow.log_metric('root_mean_squared_error', rmse)

        example_input = X_train.iloc[:5]
        example_output = best_model.predict(example_input)

        signature = mlflow.models.infer_signature(
            example_input,
            example_output
        )

        mlflow.sklearn.log_model(
            best_model,
            'best_model',
            signature=signature,
            input_example=example_input
        )

        current_file = os.path.abspath(__file__)
        mlflow.log_artifact(current_file)

        mlflow.register_model(
            f'runs:/{mlflow.active_run().info.run_id}/best_model',
            'MedicalRegressor',
        )
