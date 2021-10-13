#!/usr/bin/env python
# coding: utf-8

# # Investigating Rewards and Loss Systems

# There are 3 payout schemes that are implemented in each study, and this notebook intends to see if this is reflected in and/or has an influnce on the subjects' choices.

# ## Importing Relevant Packages and Datasets

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functions import *


# In[5]:


index_95 = pd.DataFrame(pd.read_csv('../data/index_95.csv'))
choice_95 = pd.DataFrame(pd.read_csv('../data/choice_95.csv'))
win_95 = pd.DataFrame(pd.read_csv('../data/wi_95.csv'))
loss_95 = pd.DataFrame(pd.read_csv('../data/lo_95.csv'))

index_100 = pd.DataFrame(pd.read_csv('../data/index_100.csv'))
choice_100 = pd.DataFrame(pd.read_csv('../data/choice_100.csv'))
win_100 = pd.DataFrame(pd.read_csv('../data/wi_100.csv'))
loss_100 = pd.DataFrame(pd.read_csv('../data/lo_100.csv'))

index_150 = pd.DataFrame(pd.read_csv('../data/index_150.csv'))
choice_150 = pd.DataFrame(pd.read_csv('../data/choice_150.csv'))
win_150 = pd.DataFrame(pd.read_csv('../data/wi_150.csv'))
loss_150 = pd.DataFrame(pd.read_csv('../data/lo_150.csv'))


# ## Creating Separate Datasets per Study

# In[9]:


## Payoff 1
study_fridberg = index_95
study_maia = index_100.iloc[181:221]
study_worthy = index_100.iloc[469:]

# Payoff 2
study_hortsmann = index_100.iloc[:162]
study_kjome = index_100.iloc[162:181]
study_steingrover_inprep = index_100.iloc[221:291]
study_premkumar = index_100.iloc[291:316]
study_steingrover_2011 = index_150.iloc[:57]
study_wetzels = index_150.iloc[57:]

# Payoff 3
study_wood = index_100.iloc[316:469]


# In[12]:


## Payoff 1
choice_fridberg = choice_95
choice_maia = choice_100.iloc[181:221]
choice_worthy = choice_100.iloc[469:]

# Payoff 2
choice_hortsmann = choice_100.iloc[:162]
choice_kjome = choice_100.iloc[162:181]
choice_steingrover_inprep = choice_100.iloc[221:291]
choice_premkumar = choice_100.iloc[291:316]
choice_steingrover_2011 = choice_150.iloc[:57]
choice_wetzels = choice_150.iloc[57:]

# Payoff 3
choice_wood = choice_100.iloc[316:469]


# In[10]:


# Payoff 1
win_fridberg = win_95
win_maia = win_100.iloc[181:221]
win_worthy = win_100.iloc[469:]

# Payoff 2
win_hortsmann = win_100.iloc[:162]
win_kjome = win_100.iloc[162:181]
win_steingrover_inprep = win_100.iloc[221:291]
win_premkumar = win_100.iloc[291:316]
win_steingrover_2011 = win_150.iloc[:57]
win_wetzels = win_150.iloc[57:]

# Payoff 3
win_wood = win_100.iloc[316:469]


# In[11]:


# Payoff 1
loss_fridberg = loss_95
loss_maia = loss_100.iloc[181:221]
loss_worthy = loss_100.iloc[469:]

# Payoff 2
loss_hortsmann = loss_100.iloc[:162]
loss_kjome = loss_100.iloc[162:181]
loss_steingrover_inprep = loss_100.iloc[221:291]
loss_premkumar = loss_100.iloc[291:316]
loss_steingrover_2011 = loss_150.iloc[:57]
loss_wetzels = loss_150.iloc[57:]

# Payoff 3
loss_wood = loss_100.iloc[316:469]


# ## Deeper Analysis of the Risk/Reward Associated with each Choice and Study

# In[15]:


## Payoff 1

times_1_chosen = 0
times_1_loss = 0
losses_1 = []
times_1_win = 0
wins_1 = []

times_2_chosen = 0
times_2_loss = 0
losses_2 = []
times_2_win = 0
wins_2 = []

times_3_chosen = 0
times_3_loss = 0
losses_3 = []
times_3_win = 0
wins_3 = []

times_4_chosen = 0
times_4_loss = 0
losses_4 = []
times_4_win = 0
wins_4 = []


for choice_win_loss_dataset in [[choice_fridberg, win_fridberg, loss_fridberg],
                          [choice_maia, win_maia, loss_maia],
                          [choice_worthy, win_worthy, win_worthy]]:
    choice_df = choice_win_loss_dataset[0]
    win_df = choice_win_loss_dataset[1]
    loss_df = choice_win_loss_dataset[2]
    num_subjects = len(choice_df.iloc[:,0])
    num_trials = len(choice_df.iloc[0])
    for subject in range(0,num_subjects):
        for round in range(0,num_trials):
            choice_made = choice_df.iloc[subject][round]
            if choice_made == 1:
                times_1_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_1_loss +=1
                    losses_1.append(loss)
                if win > 0:
                    times_1_win += 1
                    wins_1.append(win)
            elif choice_made == 2:
                times_2_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_2_loss +=1
                    losses_2.append(loss)
                if win > 0:
                    times_2_win += 1
                    wins_2.append(win)
            elif choice_made == 3:
                times_3_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_3_loss +=1
                    losses_3.append(loss)
                if win > 0:
                    times_3_win += 1
                    wins_3.append(win)
            elif choice_made == 4:
                times_4_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_4_loss +=1
                    losses_4.append(loss)
                if win > 0:
                    times_4_win += 1
                    wins_4.append(win)

print("Times 1 was chosen=", times_1_chosen)
print("Times 1 was a loss=", times_1_loss)
print("Times 1 was a win=", times_1_win)

print("Times 2 was chosen=", times_2_chosen)
print("Times 2 was a loss=", times_2_loss)
print("Times 2 was a win=", times_2_win)

print("Times 3 was chosen=", times_3_chosen)
print("Times 3 was a loss=", times_3_loss)
print("Times 3 was a win=", times_3_win)

print("Times 4 was chosen=", times_4_chosen)
print("Times 4 was a loss=", times_4_loss)
print("Times 4 was a win=", times_4_win)


# In[16]:


## Payoff 2

times_1_chosen = 0
times_1_loss = 0
losses_1 = []
times_1_win = 0
wins_1 = []

times_2_chosen = 0
times_2_loss = 0
losses_2 = []
times_2_win = 0
wins_2 = []

times_3_chosen = 0
times_3_loss = 0
losses_3 = []
times_3_win = 0
wins_3 = []

times_4_chosen = 0
times_4_loss = 0
losses_4 = []
times_4_win = 0
wins_4 = [] 

for choice_win_loss_dataset in [[choice_hortsmann, win_hortsmann, loss_hortsmann],
                          [choice_kjome, win_kjome, loss_kjome],
                          [choice_steingrover_inprep, win_steingrover_inprep, win_steingrover_inprep],
                          [choice_premkumar, win_premkumar, loss_premkumar],
                          [choice_steingrover_2011, win_steingrover_2011, win_steingrover_2011],
                          [choice_wetzels, win_wetzels, loss_wetzels],]:
    choice_df = choice_win_loss_dataset[0]
    win_df = choice_win_loss_dataset[1]
    loss_df = choice_win_loss_dataset[2]
    num_subjects = len(choice_df.iloc[:,0])
    num_trials = len(choice_df.iloc[0])
    for subject in range(0,num_subjects):
        for round in range(0,num_trials):
            choice_made = choice_df.iloc[subject][round]
            if choice_made == 1:
                times_1_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_1_loss +=1
                    losses_1.append(loss)
                if win > 0:
                    times_1_win += 1
                    wins_1.append(win)
            elif choice_made == 2:
                times_2_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_2_loss +=1
                    losses_2.append(loss)
                if win > 0:
                    times_2_win += 1
                    wins_2.append(win)
            elif choice_made == 3:
                times_3_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_3_loss +=1
                    losses_3.append(loss)
                if win > 0:
                    times_3_win += 1
                    wins_3.append(win)
            elif choice_made == 4:
                times_4_chosen += 1
                loss = loss_df.iloc[subject][round]
                win = win_df.iloc[subject][round]
                if loss < 0:
                    times_4_loss +=1
                    losses_4.append(loss)
                if win > 0:
                    times_4_win += 1
                    wins_4.append(win)

print("Times 1 was chosen=", times_1_chosen)
print("Times 1 was a loss=", times_1_loss)
print("Times 1 was a win=", times_1_win)

print("Times 2 was chosen=", times_2_chosen)
print("Times 2 was a loss=", times_2_loss)
print("Times 2 was a win=", times_2_win)

print("Times 3 was chosen=", times_3_chosen)
print("Times 3 was a loss=", times_3_loss)
print("Times 3 was a win=", times_3_win)

print("Times 4 was chosen=", times_4_chosen)
print("Times 4 was a loss=", times_4_loss)
print("Times 4 was a win=", times_4_win)


# In[17]:


## Payoff 1

times_1_chosen = 0
times_1_loss = 0
losses_1 = []
times_1_win = 0
wins_1 = []

times_2_chosen = 0
times_2_loss = 0
losses_2 = []
times_2_win = 0
wins_2 = []

times_3_chosen = 0
times_3_loss = 0
losses_3 = []
times_3_win = 0
wins_3 = []

times_4_chosen = 0
times_4_loss = 0
losses_4 = []
times_4_win = 0
wins_4 = []

choice_df = choice_wood
win_df = win_wood
loss_df = loss_wood
num_subjects = len(choice_df.iloc[:,0])
num_trials = len(choice_df.iloc[0])
for subject in range(0,num_subjects):
    for round in range(0,num_trials):
        choice_made = choice_df.iloc[subject][round]
        if choice_made == 1:
            times_1_chosen += 1
            loss = loss_df.iloc[subject][round]
            win = win_df.iloc[subject][round]
            if loss < 0:
                times_1_loss +=1
                losses_1.append(loss)
            if win > 0:
                times_1_win += 1
                wins_1.append(win)
        elif choice_made == 2:
            times_2_chosen += 1
            loss = loss_df.iloc[subject][round]
            win = win_df.iloc[subject][round]
            if loss < 0:
                times_2_loss +=1
                losses_2.append(loss)
            if win > 0:
                times_2_win += 1
                wins_2.append(win)
        elif choice_made == 3:
            times_3_chosen += 1
            loss = loss_df.iloc[subject][round]
            win = win_df.iloc[subject][round]
            if loss < 0:
                times_3_loss +=1
                losses_3.append(loss)
            if win > 0:
                times_3_win += 1
                wins_3.append(win)
        elif choice_made == 4:
            times_4_chosen += 1
            loss = loss_df.iloc[subject][round]
            win = win_df.iloc[subject][round]
            if loss < 0:
                times_4_loss +=1
                losses_4.append(loss)
            if win > 0:
                times_4_win += 1
                wins_4.append(win)

print("Times 1 was chosen=", times_1_chosen)
print("Times 1 was a loss=", times_1_loss)
print("Times 1 was a win=", times_1_win)

print("Times 2 was chosen=", times_2_chosen)
print("Times 2 was a loss=", times_2_loss)
print("Times 2 was a win=", times_2_win)

print("Times 3 was chosen=", times_3_chosen)
print("Times 3 was a loss=", times_3_loss)
print("Times 3 was a win=", times_3_win)

print("Times 4 was chosen=", times_4_chosen)
print("Times 4 was a loss=", times_4_loss)
print("Times 4 was a win=", times_4_win)


# ## Creating graphs to see differences

# In[19]:


# Risk of loss per choice per study

payoff1_risk = [383/1400,
               153/2516,
               694/2232,
               169/2777]
payoff2_risk = [2039/5793,
               851/13683,
               3182/10485,
               825/12339]
payoff3_risk = [1402/2564,
               490/4402,
               2029/3523,
               458/4811]
labels = [1, 2, 3, 4]


# In[29]:


fig, ax = plt.subplots()
x = np.arange(len(labels))  # the label locations
width = 0.25

rects1 = ax.bar(x, payoff1_risk, width, label='Payoff 1')
rects2 = ax.bar(x +.25, payoff2_risk, width, label='Payoff 2')
rects2 = ax.bar(x + .5, payoff3_risk, width, label='Payoff 3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Chance of receiving a loss')
ax.set_xlabel('Choices')
ax.set_title('Risk of Receiving a Loss per Choice per Payoff Scheme')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()


# As shown in the above graph and indicated in [this paper](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/), the decks 1 and 3 have more frequent penalties for its subjects. All three payoff schemes seem to be very similar.
# 
# In payoff scheme 1, deck 3 holds a higher penalty risk than deck 1, although from the graph we can see that for payoff scheme 2, it is riskier to choose from 1. 

# In[ ]:




