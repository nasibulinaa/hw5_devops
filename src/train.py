import shutil
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

def mlflow_experiment(df: pd.DataFrame):
    mlflow.set_experiment("iris") # creates an experiment if it doesn't exist

    x = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    with mlflow.start_run(run_name="Iris RF Experiment") as run:

        # add parameters for tuning
        num_estimators = 100
        mlflow.log_param("num_estimators", num_estimators)

        # train the model
        model_rf = RandomForestRegressor(n_estimators=num_estimators)
        model_rf.fit(X_train, y_train)
        predictions = model_rf.predict(X_test)

        # save the model artifact for deployment
        # this will save the model locally or to the S3 bucket if using a server
        mlflow.sklearn.log_model(sk_model=model_rf,
                                 name='models',
                                 input_example = x.iloc[:3])
        if os.path.exists("models/model"):
            shutil.rmtree("models/model")
        mlflow.sklearn.save_model(model_rf, "models/model")
        #model_uri = "runs:/{}/model".format(run.info.run_id)

        # log model performance 
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)
        print("  mse: %f" % mse)
        print(mlflow.get_artifact_uri())
        print("runID: %s" % run.info.run_id)
