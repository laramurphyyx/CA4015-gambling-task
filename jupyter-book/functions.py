import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# +
## Cleaning

def check_for_null_values(dataframe):
    count_null_values = dataframe.isnull().values.sum()
    assertion_error = "There were " + str(count_null_values) + " null values found."
    assert dataframe.isnull().values.any() == False, assertion_error
            
def check_for_outlier(dataframe, list):
    for subject in dataframe.values:
        for choice in subject:
            assertion_error = "There is an outlier: " + str(choice)
            assert int(choice) in list, assertion_error

def check_all_data_type(dataframe,data_type):
    for subject in dataframe.values:
        for choice in subject:
            assertion_error = str(type(choice)) + " is not of class " + str(data_type)
            assert type(choice) == data_type, assertion_error

def check_for_duplicate_rows(dataframe):
    duplicates = 0
    for row in dataframe.duplicated():
        if row == True:
            duplicates += 1
    print("There were " + str(duplicates) + " duplicates found.")
# +
## Processing and Analysing

def create_int_column_name(column):
    return int(column.split("_")[1])

def change_columns_to_int(dataframe):
    dataframe.rename(columns=lambda x: create_int_column_name(x), inplace=True)

def create_plottable_array(dataframe):
    tmp_list = []
    num_subjects = len(dataframe.iloc[:][1])
    num_trials = len(dataframe.columns)
    for subject in range(0,num_subjects):
        for trial in range(0,num_trials):
            tmp_list.append([dataframe.columns[trial],int(dataframe.iloc[subject][trial + 1])])
    return np.array(tmp_list)

def create_running_average_array(dataframe, row):
    avg_data_row = [dataframe.iloc[0][1]]
    tmp_list = []
    for i in range(1,95):
        rolling_average = ((avg_data_row[-1]*(i)) + dataframe.iloc[row][i+1]) / (i+1)
        avg_data_row.append(rolling_average)
    for i in range(0,len(avg_data_row)):
        tmp_list.append([dataframe.columns[i],avg_data_row[i]])
    my_array = np.array(tmp_list)
    return my_array

