import pandas as pd
import argparse
import pickle
import warnings

warnings.filterwarnings(action='ignore')
RANDOM_STATE = 42

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train_path', required=True, help='Path to your training data')
arg_parser.add_argument('--test_path', required=True, help='Path to your test data')
arg_parser.add_argument('--train_sample', required=True, help='An integer to sample your training data (ex: 5 to sample 5% of training data)')
arg_parser.add_argument('--test_sample', required=True, help='An integer to sample your test data (ex: 5 to sample 5% of test data)')
arg_parser.add_argument('--train_save_path', required=True, help='Path to save the sampled training data with .pickle as extension (ex: Data/sampled_train.pickle)')
arg_parser.add_argument('--test_save_path', required=True, help='Path to save the sampled test data .pickle as extension (ex: Data/sampled_train.pickle)')
args = vars(arg_parser.parse_args())

# load data
train_full = pd.read_csv(args['train_path'])
test_full = pd.read_csv(args['test_path'])
print('[INFO] Found {} rows of training data and {} rows of test data'.format(len(train_full), len(test_full)))
# sample both train & test data
sampled_train = train_full.sample(frac=float(args['train_sample'])/100, random_state=RANDOM_STATE)
sampled_test = test_full.sample(frac=float(args['test_sample'])/100, random_state=RANDOM_STATE)
print('[INFO] You have sampled {} rows of training data and {} rows of test data'.format(len(sampled_train), len(sampled_test)))

# save both sampled data
print('[INFO] Saving both sampled data...')
with open(args['train_save_path'], 'wb') as f:
    pickle.dump(sampled_train, f)
with open(args['test_save_path'], 'wb') as f:
    pickle.dump(sampled_test, f)
print('[INFO] Saving finished, exiting program...')