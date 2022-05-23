import pandas as pd
import pickle

def load_data(path):
    if '.csv' in path:
        data = pd.read_csv(path)
    elif '.pickle' in path or '.pkl' in path:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            data = data['data']

    return data

def create_vector_space(df, features='vendor_tag_name', is_vendor=False):
    df[features] = df[features].fillna('None')
    df[features] = df[features].apply(lambda x: x.split(','))
    vector_space = pd.get_dummies(df[features].apply(pd.Series).stack()).sum(level=0)
    if is_vendor:   
        vector_space.index = df['id']
    else:
        vector_space.index = df['customer_id']
        vector_space = vector_space.groupby(vector_space.index).sum()

    return vector_space


