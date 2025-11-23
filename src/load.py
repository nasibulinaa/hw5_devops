import os
import pandas as pd
from sklearn.datasets import load_iris

DATA_LOCATION='data/iris.csv'

def load_and_prepare_data():
    if os.path.isfile(DATA_LOCATION):
        df = pd.read_csv(DATA_LOCATION)
    else:
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df.to_csv('data/iris.csv', index=False)
    return df