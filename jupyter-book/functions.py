import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# +
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
# -


