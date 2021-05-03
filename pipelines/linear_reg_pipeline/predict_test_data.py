import os
import argparse
import joblib
from azureml.core import Run, Model

import pandas as pd
from sklearn.linear_model import LinearRegression

from transformation_sub_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--scaled-test-data", type=str, dest = 'scaled_data', help='Scaled test dataset')
parser.add_argument('--trained-data', type=str, dest = 'model', help='Trained model')
parser.add_argument('--predicted-test-data', type=str, dest = 'predictions', help='Folder for predictions')

args = parser.parse_args()
run = Run.get_context()

print("Loading Data...")
file_path = os.path.join(args.scaled_data,'scaled_test.csv')
x_data = pd.read_csv(file_path)

model_path = Model.get_model_path('model.pkl')
model = joblib.load(model_path)

y_hat_train = model.predict(x_data)

run.log('Test prediction: ', str(y_hat_train))

print("Saving Predictions...")
save_path = os.path.join(args.predictions,'predictions.csv')
y_hat_train.to_csv(save_path, header=True)

run.complete()