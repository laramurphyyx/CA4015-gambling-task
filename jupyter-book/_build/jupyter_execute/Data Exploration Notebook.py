#!/usr/bin/env python
# coding: utf-8

# # Exploratory Work

# ## Importing Relevant Packages and Datasets

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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


# In[3]:


datasets = [choice_95, win_95, loss_95, 
          choice_100, win_100, loss_100, 
          choice_150, win_150, loss_150]

for dataset in datasets:
    change_columns_to_int(dataset)


# Changing the column names from strings to integers allows for plotting on a graph

# ## Popularity of each choice throughout
# 
# We know that there are certain decks that yield a higher return than others, but the subjects participating in the task do not.
# 
# The following code snippets will be exploring the popularity of each choice throughout the game.

# In[4]:


array_95 = create_plottable_array(choice_95)
array_100_pt1 = create_plottable_array(choice_100.iloc[:168])
array_100_pt2 = create_plottable_array(choice_100.iloc[168:336])
array_100_pt3 = create_plottable_array(choice_100.iloc[336:])
array_150 = create_plottable_array(choice_150)


# As there were 504 subjects contained in the 'array_100' dataset, they have been split into three groups to allow for clearer analysis of the graphs

# In[5]:


# plot the data points
plt.scatter(
   array_95[:, 0], array_95[:, 1],
   c='blue', alpha=0.03,
   edgecolor='black'
)


# In[6]:


# plot the data points
plt.scatter(
   array_100_pt1[:, 0], array_100_pt1[:, 1],
   c='blue', alpha=0.002,
   edgecolor='black'
)


# In[7]:


# plot the data points
plt.scatter(
   array_100_pt2[:, 0], array_100_pt2[:, 1],
   c='blue', alpha=0.002,
   edgecolor='black'
)


# In[8]:


# plot the data points
plt.scatter(
   array_100_pt3[:, 0], array_100_pt3[:, 1],
   c='blue', alpha=0.002,
   edgecolor='black'
)


# In[9]:


# plot the data points
plt.scatter(
   array_150[:, 0], array_150[:, 1],
   c='blue', alpha=0.005,
   edgecolor='black'
)


# The above graphs are very similar, this implies that the popularity of choices is independent of the number of trials the participant played.
# 
# Choice 1 seems to be at it's most popular at the beginning of the game, but is much less likely to be chosen as the trials continue.
# 
# Choice 2 is very popular at the beginning, but decreases slightly towards the end.
# 
# Choice 3 is not very popular at the beginning, but gains some popularity towards the end of the task.
# 
# Choice 4 is consistently popular throughout the entire task.

# ## Viewing the running average of a selected group of subjects

# In[10]:


running_average_95 = create_running_average_array(choice_95, 1)
running_average_100 = create_running_average_array(choice_100, 452)
running_average_150 = create_running_average_array(choice_150, 83)


# In[11]:


plt.scatter(
   running_average_95[:, 0], running_average_95[:, 1],
    s=25, alpha=0.3, c='blue',
    marker='o',
    label='avg_95'
)

plt.scatter(
   running_average_100[:, 0], running_average_100[:, 1],
    s=25, alpha=0.2, c='orange',
    marker='o',
    label='avg_100'
)

plt.scatter(
   running_average_150[:, 0], running_average_150[:, 1],
    s=25, alpha=0.4, c='yellow',
    marker='o',
    label='avg_150'
)

plt.title("The Total Profit/Loss per Subject per Choice")
plt.xlabel("Total Profit/Loss")
plt.ylabel("Number of times the option was chosen")
plt.legend(scatterpoints=1)


# The graphs above are not representative of the entire group of participants, as plotting the choices of 617 people on a chart would not be legible.
# 
# The few selected subjects are showing an expected scattered plot at the beginning of the task, with the direction of the plot becoming narrower towards the end.
# 
# These figures will not be useful for clustering as the influence of the choice on the running average becomes smaller and smaller and is not hugely representative of their choices towards the end, although it is an interesting observation.
