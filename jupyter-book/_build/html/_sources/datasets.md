# About the Datasets

## Dataset Origins

The sources of these 9 datasets can be found in [this paper](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/). The dataset contains results obtained by 10 different studies. These results were collected between the years of 2004 and 2015.


## Dataset Contents

There are four values being measured in the datasets, the choice each subject made each trial, the reward value for making that choice, the penalty cost for making that choice and the study that they were a part of. All of the studies either have 95, 100 or 150 trials, so there are 4 datasets for each study size and for each value being measures, giving 12 datasets.

The datasets are formatted as follows:
- **choice_{#trials}**: The dataset containing the choices of subjects who played {#trials} trials 
- **wi_{#trials}**: The dataset containing the reward values of subjects who played {#trials} trials 
- **lo_{#trials}**: The dataset containing the losses of subjects who played {#trials} trials 
- **index_{#trials}**: The dataset containing the studies that the subjects who played {#trials} trials participated in


## Dataset Structure

Each dataset follows the same structural layout. All of the datasets have named columns and rows. The column names of each dataset are the same depending on the value being measured. For example, all three datasets that measure the winning values (wi_95, wi_100, wi_150) all have the naming convention 'Wins_{trial}' for columns. Similarly, the naming conventions for the choices, losses and study are 'Choice_{trial}', 'Losses_{trial}' and 'Study' respectively.

There are a total of 617 subjects accounted for in these datasets. 15 subjects played 95 trials, 504 subjects played 100 trials and 98 played 150 trials.