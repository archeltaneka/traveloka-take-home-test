import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
import ml_metrics as metrics
from haversine import haversine


def train_model(df_profile, n_recommendations=20, metric='cosine', algorithm='brute'):
    n_recommendations = 20
    knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_recommendations)
    knn.fit(df_profile.values)
    return knn

def evaluate_model(knn, df_train_or_test, df_vendor, vendor_profile, user_profile):
    user_locations = df_train_or_test[['id', 'latitude_y', 'longitude_y', 'customer_id']]
    user_locations = user_locations[user_locations['customer_id'].notnull()] # filter data with null customer_id

    total_apk = 0
    for _, user_row in user_locations.iterrows():
        _, vendor_idx = knn.kneighbors(np.array(user_profile.loc[user_row['customer_id']]).reshape(1,-1))
        recommended_vendor_ids = vendor_profile.iloc[vendor_idx[0]].index
        recommended_vendors = df_vendor[df_vendor['id'].isin(recommended_vendor_ids)]
        distance_from_user = []
        for _, row in recommended_vendors.iterrows():
            dist = haversine([user_row['latitude_y'], user_row['longitude_y']], [row['latitude'], row['longitude']])
            distance_from_user.append(dist)
        recommended_vendors['distance_from_user'] = distance_from_user
        recommended_vendors = recommended_vendors.sort_values(by=['distance_from_user'])
        total_apk += metrics.apk([user_row.id], recommended_vendors.id.tolist(), len(recommended_vendors))

    return total_apk/len(user_locations)

def recommend_a_user(knn, user_id, df_train, df_vendor, user_profile, vendor_profile):
    selected_user = df_train[df_train['customer_id']==user_id]
    if len(selected_user) > 1:
        selected_user = selected_user.iloc[0]

    _, vendor_idx = knn.kneighbors(np.array(user_profile.loc[user_id]).reshape(1,-1))
    recommended_vendor_ids = vendor_profile.iloc[vendor_idx[0]].index
    recommended_vendors = df_vendor[df_vendor['id'].isin(recommended_vendor_ids)]
    distance_from_user = []
    for _, row in recommended_vendors.iterrows():
        dist = haversine([selected_user['latitude_y'], selected_user['longitude_y']], [row['latitude'], row['longitude']])
        distance_from_user.append(dist)
    recommended_vendors['distance_from_user'] = distance_from_user
    recommended_vendors = recommended_vendors.sort_values(by=['distance_from_user'])

    return recommended_vendors