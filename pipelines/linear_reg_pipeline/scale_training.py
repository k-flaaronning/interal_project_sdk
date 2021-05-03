import os
import argparse
import joblib
from azureml.core import Run

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from transformation_sub_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, help='Training dataset')
parser.add_argument('--scaled-train-data', type=str, dest = 'scaled_train', help='Scaled training dataset output')

args = parser.parse_args()
run = Run.get_context()

print("Loading Data...")
train = run.input_datasets['feature_selected_train_FD001'].to_pandas_dataframe()

index_names = train.columns[[0, 1]]
setting_names = train.columns[[2]]
sensor_names = train.drop(index_names.union(setting_names), axis = 1).columns # Find something better than union!!
scale_columns = sensor_names
keep_columns = scale_columns.union(index_names[[1]])

x_train = add_remaining_useful_life(train)
scaler = MinMaxScaler().fit(x_train[keep_columns])
x_train_scaled = x_train.copy()
x_train_scaled[keep_columns] = pd.DataFrame(scaler.transform(x_train[keep_columns]))

print("Saving Data...")
os.makedirs(args.scaled_train, exist_ok=True)
save_path = os.path.join(args.scaled_train,'scaled_train.csv')
x_train_scaled.to_csv(save_path, header=True)

print("Saving Scaler...")
scaler_file = os.path.join(args.scaled_train,'scaler.save')
joblib.dump(value=scaler, filename=scaler_file)

run.complete()