import mlflow
from prefect import task, flow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5001")

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


@task
def preprocess_data(df):
    print(df.head())
    df = df[["PULocationID", "DOLocationID", "duration"]]
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(str)
    return df


@task
def generate_vectorizer(df):
    train_dicts = df[["PULocationID", "DOLocationID"]].to_dict(orient='records')
    vectorizer = DictVectorizer()
    vectorizer.fit(train_dicts)
    return train_dicts, vectorizer

@task
def transform_data(data_dict, vectorizer):
    feature_mat = vectorizer.transform(data_dict)
    return feature_mat


@task(log_prints=True)
def train_model(X_train, y_train):
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        #print(f"model intercept: {model.intercept_}")
        mlflow.log_param("model_intercept", model.intercept_)
        #mlflow.log_metric("metric_name", metric_value)
        mlflow.sklearn.log_model(model, "LR_model")
    return model


@task
def predict(model, X_test):
    return model.predict(X_test)


@task
def calculate_rmse(y_test, y_pred):
    return root_mean_squared_error(y_test, y_pred)


@flow
def taxi_duration_predictor():
    df = read_dataframe("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet")
    df = preprocess_data(df)
    data_dict, vectorizer = generate_vectorizer(df)
    
    X_train = transform_data(data_dict, vectorizer)
    target = 'duration'
    y_train = df[target].values
    
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_train) # Test on train data
    rmse = calculate_rmse(y_train, y_pred)

# flow executor
if __name__ == "__main__":
    taxi_duration_predictor()
