#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import argparse

def read_data(year, month):

    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(filename)

    categorical = ['PULocationID', 'DOLocationID']

    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def generate_results(df, year, month):
    
    categorical = ['PULocationID', 'DOLocationID']
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['prediction'] = y_pred
    df_result = df[['ride_id', 'prediction']]

    print(np.mean(y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download monthly data file & run predict()")
    parser.add_argument("--year", type=int, required=True, help="Year of the data (e.g. 2023)")
    parser.add_argument("--month", type=int, required=True, help="Month of the data (1-12)")

    args = parser.parse_args()
    df = read_data(args.year, args.month)
    generate_results(df, args.year, args.month)
