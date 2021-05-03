import pandas as pd
import numpy as np

def generator_training(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]
        
def generator_training_wrapper(df, sequence_length, columns, unit_nos=np.array([])):
    if unit_nos.size <= 0:
        unit_nos = df['unit_no'].unique()
    #Runs the generator_training_data function for all units
    seperate_unit_gen = (list(generator_training(df[df['unit_no']==unit_no], sequence_length, columns))
               for unit_no in unit_nos)
    #Combine the subsets into a new set of sequences    
    combined_units_gen = np.concatenate(list(seperate_unit_gen)).astype(np.float32)
    return combined_units_gen

def generator_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # padded sequences
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:,:] = df[columns].values
    else:
        data_matrix = df[columns].values
        
    stop = num_elements = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]

def generator_test_wrapper(df, sequence_length, columns, unit_nos=np.array([])):
    if unit_nos.size <= 0:
        unit_nos = df['unit_no'].unique()
    test_gen = (list(generator_test_data(df[df['unit_no']==unit_no], sequence_length, columns, -99.))
           for unit_no in unit_nos)   
    combined_units_gen = np.concatenate(list(test_gen)).astype(np.float32)
    return combined_units_gen

def generator_labels(df, sequence_length, label_column):
    data = df[label_column].values
    num_elements = data.shape[0]
    #-1 makes sure that the label returned is the last row of the sequence, not the beginning of the next sequence
    return data[sequence_length-1:num_elements, :]

def generator_label_wrapper(df, sequence_length, label, unit_nos=np.array([])):
    if unit_nos.size <= 0:
        unit_nos = df['unit_no'].unique()
        
    seperate_unit_gen = (generator_labels(df[df['unit_no']==unit_no], sequence_length, label) 
                for unit_no in unit_nos)
    comined_units_gen = np.concatenate(list(seperate_unit_gen)).astype(np.float32)
    return comined_units_gen