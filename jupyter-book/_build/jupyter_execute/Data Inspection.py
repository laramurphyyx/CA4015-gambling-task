#!/usr/bin/env python
# coding: utf-8

# # Data Inspection

# This is an initial inspection of the datasets. This will ensure that the data will be ready to used for clustering.

# ## 1.    Importing relevant packages and Datasets

# In[1]:


import pandas as pd
import numpy as np
from functions import *


# In[2]:


choice_95 = pd.DataFrame(pd.read_csv('../data/choice_95.csv'))
win_95 = pd.DataFrame(pd.read_csv('../data/wi_95.csv'))
loss_95 = pd.DataFrame(pd.read_csv('../data/lo_95.csv'))

choice_100 = pd.DataFrame(pd.read_csv('../data/choice_100.csv'))
win_100 = pd.DataFrame(pd.read_csv('../data/wi_100.csv'))
loss_100 = pd.DataFrame(pd.read_csv('../data/lo_100.csv'))

choice_150 = pd.DataFrame(pd.read_csv('../data/choice_150.csv'))
win_150 = pd.DataFrame(pd.read_csv('../data/wi_150.csv'))
loss_150 = pd.DataFrame(pd.read_csv('../data/lo_150.csv'))


# ## 2.    Ensuring Data is Clean

# It is important to check that the data is clean so it can be processed correctly for analysis.
# 
# This includes checking that the data is accurate, that there are no structural errors in the data, that there are no null values and investigating outliers or duplicate values.
# 
# The data seems to be reliable, as they are all results from studies performed in the papers listed [here](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/). 
# 
# All of the datasets seem to be inherent of identical structures, with 'Choice_{number}', 'Wins_{number}' and 'Losses_{number}' as the column names for the choices, wins and losses datasets respectively. The row names in all datasets follow the same structure also, being 'Subj_{number}'
# 
# The code snippets below examine the datasets for outliers and duplicate rows and ensure they are all the same type.

# In[3]:


datasets = [choice_95, win_95, loss_95,
            win_100, loss_100, 
           choice_150, win_150, loss_150,
           choice_100]

for dataset in datasets:
    check_for_null_values(dataset)
    check_for_duplicate_rows(dataset)


# The check_for_null_values() function uses an assertion, so as there are no errors in the output, it shows that all datasets contain non-null values only.
# 
# The check_for_duplicate_rows() function checks the quantity of duplicate rows. This function does not use an assertion, as it is entirely possible that two subjects participating in the task may have received the same sequence of rewards or penalties. The validity of this function is subjective, but I do not believe that there are any mistaken duplicates in these datasets.

# In[ ]:


check_for_outlier(choice_95, [1,2,3,4])
check_for_outlier(choice_100, [1,2,3,4])
check_for_outlier(choice_150, [1,2,3,4])

check_for_outlier(win_95, range(0,200))
check_for_outlier(win_100, range(0,200))
check_for_outlier(win_150, range(0,200))

check_for_outlier(loss_95, range(-3000, 1))
check_for_outlier(loss_100, range(-3000, 1))
check_for_outlier(loss_150, range(-3000, 1))


# In[ ]:


check_all_data_type(choice_95, np.int64)
check_all_data_type(choice_100, np.int64)
check_all_data_type(choice_150, np.int64)

check_all_data_type(win_95, np.int64)
check_all_data_type(win_100, np.int64)
check_all_data_type(win_150, np.int64)

check_all_data_type(loss_95, np.int64)
check_all_data_type(loss_100, np.int64)
check_all_data_type(loss_150, np.int64)


# Both of the functions above, 'check_for_outlier()' and 'check_all_data_types()', both use assertion statements. No output shows us that all datasets are clear from outliers or mismatched data types.
# 
# The datasets seem to be clean, with no inconsistent data types or structural differences, and no obvious outliers or inaccuracies.
