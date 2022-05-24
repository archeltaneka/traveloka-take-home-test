import pandas as pd
import argparse
import pickle
from datetime import datetime

from utils import create_vector_space, load_data
from model_utils import train_model, evaluate_model, recommend_a_user

import warnings
warnings.filterwarnings(action='ignore')


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train_path', required=True, help='Full or sampled training data (currently compatible with .csv and .pkl/.pickle extension files)')
arg_parser.add_argument('--test_path', required=True, help='Full or sampled test data (currently compatible with .csv and .pkl/.pickle extension files)')
arg_parser.add_argument('--vendor_path', required=True, help='Vendor/restaurant data (currently compatible with .csv and .pkl/.pickle extension files)')
arg_parser.add_argument('--algorithm', required=False, default='brute', help='KNN algorithm that will be used (default is "brute")')
arg_parser.add_argument('--metric', required=False, default='cosine', help='KNN metric that will be used (default is "cosine")')
arg_parser.add_argument('--n_recommendations', required=False, default=20, help='Number of recommendations that KNN will provide (default is 20)')
arg_parser.add_argument('--save_model_as', required=False, help='Save the model in a directory')
arg_parser.add_argument('--save_vendor_profile', required=True, choices=['yes', 'no'], help='Whether to save vendor profiles in a directory')
arg_parser.add_argument('--save_user_profile', required=True, choices=['yes', 'no'], help='Whether to save user profiles in a directory')
args = vars(arg_parser.parse_args())

# load data
vendors = load_data(args['vendor_path'])
print('[INFO] Found {} rows of vendors data'.format(len(vendors)))
train_data = load_data(args['train_path'])
print('[INFO] Found {} rows of training data'.format(len(train_data)))
test_data = load_data(args['test_path'])
print('[INFO] Found {} rows of test data'.format(len(test_data)))

# create vector space
print('[INFO] Creating vector space for vendors, training, and test data...')
vendor_vector_space = create_vector_space(vendors, is_vendor=True)
train_vector_space = create_vector_space(train_data)
test_vector_space = create_vector_space(test_data)
combined_user_space = train_vector_space.append(test_vector_space)

# model fit
print('[INFO] Training model...')
print('[INFO] KNN Model configurations:\n k={}\n Algorithm: {}\n Metric={}'.format(args['n_recommendations'], args['algorithm'], args['metric']))
if args['n_recommendations'] is not None:
    model = train_model(vendor_vector_space, n_recommendations=int(args['n_recommendations']))
else:
    model = train_model(vendor_vector_space)

# evaluate model
print('[INFO] Evaluating model...')
train_apk = evaluate_model(model, train_data, vendors, vendor_vector_space, train_vector_space)
test_apk = evaluate_model(model, test_data, vendors, vendor_vector_space, test_vector_space)
print('[INFO] Training MAP@K: {:.2f}'.format(train_apk))
print('[INFO] Test MAP@K: {:.2f}'.format(test_apk))

# save model, vendor, and user profile
if args['save_model_as'] is not None:
    with open(args['save_model_as'], 'wb') as f:
        pickle.dump(model, f)
    print('[INFO] Model succesfully saved:', args['save_model_as'])
user_profile_fn = 'user_profile_' + str(datetime.now().date()) + '.pickle'
vendor_profile_fn = 'vendor_profile_' + str(datetime.now().date()) + '.pickle'
if args['save_vendor_profile'] == 'yes':
    with open(vendor_profile_fn, 'wb') as f:
        pickle.dump(vendor_vector_space, f)
    print('[INFO] Vendor profiles succesfully saved:', vendor_profile_fn)
if args['save_user_profile'] == 'yes':
    with open(user_profile_fn, 'wb') as f:
        pickle.dump(combined_user_space, f)
    print('[INFO] User profiles succesfully saved:', user_profile_fn)