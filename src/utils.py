import pandas as pd
import pickle

def load_data(path):
    """
    Load .csv or pickle (.pickle or .pkl) file extensions.

    Arguments:
        path: string
            file path to the data

    Returns:
        data: pd.DataFrame
            loaded data
    """

    if '.csv' in path:
        data = pd.read_csv(path)
    elif '.pickle' in path or '.pkl' in path:
        with open(path, 'rb') as f:
            data = pickle.load(f)

    return data

def create_vector_space(df, features='vendor_tag_name', is_vendor=False):
    """
    Create vector space from 'vendor_tag_name' features.

    For example, if we have three restaurants: A, B, C with each of them tagged as
        A: Burger, Pasta, Pizza
        B: Sushi, Pizza
        C: Noodle, Chinese
    The resulting vector space would be something like:
            Burger | Pasta | Pizza | Sushi | Noodle | Chinese
    A       1           1       1       0       0       0          
    B       0           0       1       1       0       0
    C       0           0       0       0       1       1

    Arguments:
        df: pd.DataFrame
            data that will be transformed into a vector space
        features: string
            column feature to transform
        is_vendor: boolean
            True if the intended data to transform is vendors/restaurants data, False otherwise
    Returns:
        vector_space: pd.DataFrame
            transformed dataframe data into a vector space
    """

    df[features] = df[features].fillna('None') # fill null/nan tags with a string 'None'
    df[features] = df[features].apply(lambda x: x.split(','))
    vector_space = pd.get_dummies(df[features].apply(pd.Series).stack()).sum(level=0) # perform one-hot encoding on the tags
    if is_vendor:   
        vector_space.index = df['id']
    else:
        vector_space.index = df['customer_id']
        vector_space = vector_space.groupby(vector_space.index).sum() # to capture historical order for each customer, group data by the customer_id, then sum the data by column

    return vector_space


