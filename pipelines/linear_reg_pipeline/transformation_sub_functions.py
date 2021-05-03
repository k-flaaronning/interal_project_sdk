import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def scale_data(train, columns):
    sc = MinMaxScaler()
    scaled_train = train.copy()
    #scaled_test = test.copy()
    scaled_train[columns] = pd.DataFrame(sc.fit_transform(scaled_train[columns]))
    #scaled_test[columns] = pd.DataFrame(sc.transform(scaled_test[columns]))
    return scaled_train

def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_no")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_no', right_index=True)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

def plot_loss(history):
    plt.figure(figsize=(13,5))
    plt.plot(range(1, len(history.history['loss'])+1), history.history['loss'], label='train')
    plt.plot(range(1, len(history.history['val_loss'])+1), history.history['val_loss'], label='validate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate(y_true, y_hat, label):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
    return rmse, variance
    
def plot_predictions(y_true, y_predicted):
    plt.figure(figsize=(13,5))
    plt.plot(y_true, label='true')
    plt.plot(y_predicted, label='predicted')
    plt.xlabel('Predictions')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()