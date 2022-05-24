import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
import ml_metrics as metrics
from haversine import haversine


def train_model(df_profile, n_recommendations=20, metric='cosine', algorithm='brute'):
    """
    Fit a KNN model.
    For more info, visit the scikit-learn's official KNN documentation https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

    Arguments:
        df_profile: pd.DataFrame
            KNN fit data
        n_recommendations: int
            number of recommendations (or parameter 'k' in KNN)
        metric: string
            KNN metric choice (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html#sklearn.metrics.DistanceMetric)
        algorithm: string
            KNN algorithm choice ('auto', 'kd_tree', 'ball_tree', or 'brute')
    
    Returns:
        knn: sklearn.neighbors._unsupervised.NearestNeighbors
            Fitted KNN model
    """

    knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_recommendations)
    knn.fit(df_profile.values)
    return knn

def evaluate_model(knn, df_train_or_test, df_vendor, vendor_profile, user_profile):
    """
    Evaluate model using the MAP@K metric.

    Arguments:
        knn: sklearn.neighbors._unsupervised.NearestNeighbors
            Fitted KNN model
        df_train_or_test: pd.DataFrame
            Data to be evaluated (train/test)
        df_vendor: pd.DataFrame
            Vendor data details
        vendor_profile: pd.DataFrame
            Vendor vector space
        user_profile: pd.DataFrame
            Customer vector space

    Returns:
        MAP@K: float
            average/mean APK value
    """

    user_locations = df_train_or_test[['id', 'latitude_y', 'longitude_y', 'customer_id']]
    user_locations = user_locations[user_locations['customer_id'].notnull()] # filter data with null customer_id

    total_apk = 0
    for _, user_row in user_locations.iterrows():
        # recommend for each customer_id
        _, vendor_idx = knn.kneighbors(np.array(user_profile.loc[user_row['customer_id']]).reshape(1,-1))
        recommended_vendor_ids = vendor_profile.iloc[vendor_idx[0]].index
        recommended_vendors = df_vendor[df_vendor['id'].isin(recommended_vendor_ids)]
        distance_from_user = []
        # calculate distance from each restaurant/vendor
        for _, row in recommended_vendors.iterrows():
            dist = haversine([user_row['latitude_y'], user_row['longitude_y']], [row['latitude'], row['longitude']])
            distance_from_user.append(dist)
        recommended_vendors['distance_from_user'] = distance_from_user
        recommended_vendors = recommended_vendors.sort_values(by=['distance_from_user']) # sort by the nearest restaurant
        total_apk += metrics.apk([user_row.id], recommended_vendors.id.tolist(), len(recommended_vendors))

    return total_apk/len(user_locations)

def recommend_a_user(knn, user_id, user_lat, user_lon, df_vendor, user_profile, vendor_profile):
    """
    Recommend nearest restaurants/vendors to a user.

    Arguments:
        knn: sklearn.neighbors._unsupervised.NearestNeighbors
            Fitted KNN model
        user_id: string
            Specific user or customer ID
        user_lat: float
            Current user's latitude location
        user_lon: float
            Current user's longitude location
        df_vendor: pd.DataFrame
            Vendor data details
        user_profile: pd.DataFrame
            Customer vector space
        vendor_profile: pd.DataFrame
            Vendor vector space
    
    Returns:
        recommended_vendors: pd.DataFrame
            Recommended nearest restaurants/vendors
    """

    _, vendor_idx = knn.kneighbors(np.array(user_profile.loc[user_id]).reshape(1,-1))
    recommended_vendor_ids = vendor_profile.iloc[vendor_idx[0]].index # find user's specific taste/preference according to historical data
    recommended_vendors = df_vendor[df_vendor['id'].isin(recommended_vendor_ids)]
    # calculate distance from each restaurant 
    distance_from_user = []
    for _, row in recommended_vendors.iterrows():
        dist = haversine([user_lat, user_lon], [row['latitude'], row['longitude']])
        distance_from_user.append(dist)
    recommended_vendors['distance_from_user'] = distance_from_user
    recommended_vendors = recommended_vendors.sort_values(by=['distance_from_user']) # sort by the nearest restaurant

    return recommended_vendors