import argparse
import pickle

from utils import load_data
from model_utils import recommend_a_user


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model_path', required=True, help='Path to the saved model')
arg_parser.add_argument('--vendor_data', required=True, help='Vendor details data')
arg_parser.add_argument('--vendor_profile', required=True, help='Saved vendor vector space')
arg_parser.add_argument('--user_data', required=True, help='Users details data')
arg_parser.add_argument('--user_profile', required=True, help='Saved users vector space')
arg_parser.add_argument('--user_id', required=True, help='User unique ID')
args = vars(arg_parser.parse_args())
print('[INFO] Creating recommendations for user {}...'.format(args['user_id']))

# load model and data
model = load_data(args['model_path'])
vendors = load_data(args['vendor_data'])
users = load_data(args['user_data'])
vendor_profile = load_data(args['vendor_profile'])
user_profile = load_data(args['user_profile'])

# predict/recommend a user
try:
    recommended_vendors = recommend_a_user(model, args['user_id'], users, vendors, user_profile, vendor_profile)
    print('[INFO] Top recommendations:\n {}'.format(recommended_vendors))
except:
    print('[INFO] User {} is not found!'.format(args['user_id']))
