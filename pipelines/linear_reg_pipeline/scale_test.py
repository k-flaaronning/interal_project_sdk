import os
import argparse
import joblib
from azureml.core import Run

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from transformation_sub_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, help='Test dataset')
parser.add_argument("--scaler", type=str, dest = 'scaler_dest', help='Scaler from training set')
parser.add_argument('--scaled-test-data', type=str, dest = 'scaled_test', help='Scaled test dataset')

args = parser.parse_args()
run = Run.get_context()

print("Loading Data...")
test = run.input_datasets['feature_selected_test_FD001'].to_pandas_dataframe()
print("Loading Scaler...")
scaler = joblib.load(os.path.join(args.scaler_dest,'scaler.save')) 

index_names = test.columns[[0, 1]]
setting_names = test.columns[[2]]
sensor_names = test.drop(index_names.union(setting_names), axis = 1).columns # Find something better than union!!
scale_columns = sensor_names
keep_columns = scale_columns.union(index_names[[1]])

x_test_scaled = test.copy()
x_test_scaled[keep_columns] = pd.DataFrame(scaler.transform(test[keep_columns]))

print("Saving Data...")
save_path = os.path.join(args.scaled_test,'scaled_test.csv')
x_test_scaled.to_csv(save_path, header=True)

run.complete()