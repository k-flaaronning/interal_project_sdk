import os
import argparse
import joblib
from azureml.core import Run, Model

import pandas as pd
from sklearn.linear_model import LinearRegression

from transformation_sub_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--scaled-train-data", type=str, dest = 'scaled_data', help='Scaled training dataset')
parser.add_argument('--trained-data', type=str, dest = 'trained_data', help='Model folder')

args = parser.parse_args()
run = Run.get_context()

print("Loading Data...")
file_path = os.path.join(args.scaled_data,'scaled_train.csv')
x_data = pd.read_csv(file_path)
y_data = x_data.pop('RUL')

reg = LinearRegression()
reg.fit(x_data, y_data)
y_hat_train = reg.predict(x_data)
rmse, r2 = evaluate(y_data, y_hat_train, 'Train')

run.log('RMSE: ', str(rmse))
run.log('R2: ', str(r2))

print("Saving Data...")
model_file = os.path.join(args.trained_data,'model.pkl')
joblib.dump(value=reg, filename=model_file)

if False:
    print('Registering model...')
    Model.register(workspace=run.experiment.workspace,
                   model_path = model_file,
                   model_name = 'lin_reg_v2',
                   tags={'Training context':'Pipeline'},
                   properties={'RMSE': np.float(rmse), 'R2': np.float(r2)})

run.complete()