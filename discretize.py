#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import numpy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


sys.argv[1] = 'dating.csv'
sys.argv[2] = 'dating-binned.csv'
#sys.argv[3] = 5
# Read the data
df = pd.read_csv(sys.argv[1])
#df = pd.read_csv('dating.csv')


# In[3]:


discrete_columns = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
all_columns = df.columns.values.tolist()
continuous_valued_columns = [item for item in all_columns if item not in discrete_columns]
print all_columns
print continuous_valued_columns


# In[4]:


(row, col) = df.shape
age_range = [18.0, 58.0]
pref_score = [0.0, 1.0]
score = [0.0, 10.0]
corr_range = [-1.00, 1.00]

#bin_N = 2
bin_N = int(sys.argv[3])
#bin_value = [i for i in range(bin_N)]
bin_seg = [1.000 * i/bin_N for i in range(0, bin_N + 1)]
#print bin_seg

age = ['age', 'age_o']
corr = ['interests_correlate']
preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']

preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',  'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']

continuous_valued_columns_bins = {}

# Segment the bins
for field in continuous_valued_columns:
    continuous_valued_columns_bins[field] = []
    if field in age:
        for i in range(0, bin_N):
            continuous_valued_columns_bins[field].append(age_range[0] + bin_seg[i] * (age_range[1] - age_range[0]))
    elif field in corr:
        for i in range(0, bin_N):
            continuous_valued_columns_bins[field].append(corr_range[0] + bin_seg[i] * (corr_range[1] - corr_range[0]))
    elif field in preference_scores_of_participant or field in preference_scores_of_partner:
        for i in range(0, bin_N):
            continuous_valued_columns_bins[field].append(pref_score[0] + bin_seg[i] * (pref_score[1] - pref_score[0]))
    else:
        for i in range(0, bin_N):
            continuous_valued_columns_bins[field].append(score[0] + bin_seg[i] * (score[1] - score[0]))

print continuous_valued_columns_bins
print len(continuous_valued_columns_bins)
#print df['pref_o_attractive'].value_counts()


# In[5]:


# Dictionary of the numbers ine ach bin
continuous_valued_columns_seg = {}
# Initalize the dict
for field in continuous_valued_columns:
    continuous_valued_columns_seg[field] = [0 for i in range(bin_N)]

for i in range(row):
    for field in continuous_valued_columns:
        # Find the bin
        for j in range(0, bin_N):
            # Corner Case
            if j == 0:
                if continuous_valued_columns_bins[field][j] <= float(df[field][i]) <= continuous_valued_columns_bins[field][j + 1]:
                    df.loc[i,field] = int(j)
                    continuous_valued_columns_seg[field][j] += 1
                    break
            elif j == bin_N - 1:
                if continuous_valued_columns_bins[field][j] < float(df[field][i]):
                    df.loc[i,field] = int(j)
                    continuous_valued_columns_seg[field][j] += 1
                    break
            else:
                if continuous_valued_columns_bins[field][j] < float(df[field][i]) <= continuous_valued_columns_bins[field][j + 1]:
                    df.loc[i,field] = int(j)
                    continuous_valued_columns_seg[field][j] += 1
                    break


# In[ ]:


df = df.astype('int64')
df.to_csv(sys.argv[2], index=False)
#df.to_csv('dating-binned.csv', index=False)
print continuous_valued_columns_seg


# In[ ]:


keylist = continuous_valued_columns_seg.keys()

for field in continuous_valued_columns:
    print str(field) + ':', continuous_valued_columns_seg[field]


# In[ ]:




