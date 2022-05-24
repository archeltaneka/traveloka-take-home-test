# Traveloka Take-home Test
Problem statement: Traveloka is now launching a service in food delivery. Given customers' locations, orders history, and restaurant information, build a recommendation system that can recommend nearest restaurants.

## Prerequisites
- Python 3.5 or above
- numpy v1.20.1
- pandas v1.2.4
- scikit-learn v1.1.1
- ml_metrics v0.1.4

## Setup
To run this project, install [Anaconda](https://www.anaconda.com/products/distribution) (or you can also use your usual Python Package Index a.k.a pip) and create a separate environment:
```
$ conda create -n myenv pip
$ conda activate myenv
$ pip install numpy pandas scikit-learn ml_metrics
```

## Usage

### Sampling Data
If you find the need to sampling any of the training and test data, you can do it by running the following command.
```
$ python sample_data.py --train_path ../Data/train_full.csv --test_path ../Data/test_full.csv --train_sample 5 --test_sample 10 --train_save_path ../Data/training_data_sampled_5_percent.pickle --test_save_path ../Data/test_data_sampled_10_percent.pickle
```

### Training
Model training is based on the KNN (K-Nearest Neighbors) algorithm. To modify the algorithm's behavior, you can specify `--algorithm`, `--metric`, and `--n_recommendations` flags. Please refer to the scikit-learn's official [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) for more info.
```
$ python train.py --train_path ../Data/training_data_sampled_5_percent.pickle --test_path ../Data/test_data_sampled_10_percent.pickle --vendor_path ../Data/vendors.csv --save_vendor_profile yes --save_user_profile yes
```

### Predict
To give recommendations to a specific user, you can run the following command. Don't forget to provide the apprppriate `user_id`.
```
$ python predict.py --model_path ../saved_models/knn.pickle --vendor_data ../Data/vendors.csv --user_data ../Data/training_data_sampled_5_percent.pickle --vendor_profile ../vendor_profiles/vendor_profile_2022-05-24.pickle --user_profile ../user_profiles/user_profile_2022-05-24.pickle --user_id LG46F74 --user_lat -0.49320 --user_lon 0.755234
```


