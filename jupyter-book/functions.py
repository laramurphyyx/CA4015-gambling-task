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

def create_net_profit_loss_list(win_df, loss_df, row):
    net_totals = []
    net_total = 0
    for trial in range(1, len(win_df.iloc[row])+1):
        winnings = win_df.iloc[row][trial]
        losses = loss_df.iloc[row][trial]
        total = winnings + losses
        net_total += total
        net_totals.append(net_total)
    return net_totals
# +
# Cluster 1: Comparing total profit/loss to number of times chosen per choice

def create_net_profit_vs_count_list(dataset, win_dataset, loss_dataset, 
                                    profit_loss = [], count = []):
    number_trials = len(dataset.iloc[0])
    number_subjects = len(dataset.iloc[:,0])
    for subject in range(0, number_subjects):
        ones_pl = 0
        twos_pl = 0
        threes_pl = 0
        fours_pl = 0
        ones_chosen = 0
        twos_chosen = 0
        threes_chosen = 0
        fours_chosen = 0
        for trial in range(0,number_trials):
            choice = dataset.iloc[subject][trial]
            win = win_dataset.iloc[subject][trial]
            loss = loss_dataset.iloc[subject][trial]
            if choice == 1:
                ones_chosen += 1
                ones_pl += (win + loss)
            elif choice == 2:
                twos_chosen += 1
                twos_pl += (win + loss)
            elif choice == 3:
                threes_chosen += 1
                threes_pl += (win + loss)
            elif choice == 4:
                fours_chosen += 1
                fours_pl += (win + loss)
        profit_loss.append([ones_pl, twos_pl, threes_pl, fours_pl])
        count.append([ones_chosen/number_trials, 
                      twos_chosen/number_trials, 
                      threes_chosen/number_trials, 
                      fours_chosen/number_trials])
    return [profit_loss, count]


# +
## Cluster 2: Comparing profits and loss madeby study

def create_avg_profit_loss_list(win_loss_datasets, existing_list=[]):
    win_dataset = win_loss_datasets[0]
    loss_dataset = win_loss_datasets[1]
    number_trials = len(win_dataset.iloc[0])
    number_subjects = len(win_dataset.iloc[:,0])
    for subject in range(0, number_subjects):
        wins = 0
        losses = 0
        for trial in range(0, number_trials):
            wins += win_dataset.iloc[subject][trial]
            losses += loss_dataset.iloc[subject][trial]
        existing_list.append([wins, losses])
    return existing_list

## Cluster 2: Comparing profits and loss madeby study

def create_avg_profit_loss__per_choice_list(win_loss_datasets, existing_list=[]):
    win_dataset = win_loss_datasets[0]
    loss_dataset = win_loss_datasets[1]
    number_trials = len(win_dataset.iloc[0])
    number_subjects = len(win_dataset.iloc[:,0])
    for subject in range(0, number_subjects):
        wins = 0
        losses = 0
        for trial in range(0, number_trials):
            wins += win_dataset.iloc[subject][trial]
            losses += loss_dataset.iloc[subject][trial]
        existing_list.append([wins/number_trials, losses/number_trials])
    return existing_list
# -


