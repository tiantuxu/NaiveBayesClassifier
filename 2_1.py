#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# In[2]:


# Read the data
df = pd.read_csv('dating.csv')


# In[3]:


(row, col) = df.shape
m = [1]
f = [0]
# Split the dataframe by gender
male = []
female = []

male = df[df['gender'].isin(m)]
female = df[df['gender'].isin(f)]


# In[4]:


preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']

N = 6
male_means = []
female_means = []

fig, ax = plt.subplots()
width = 1.0
ind = np.array([0,3,6,9,12,15])

for pref in preference_scores_of_participant:
    male_means.append(male[pref].mean())
    female_means.append(female[pref].mean())
    
p1 = ax.bar(ind, male_means, width, color='red')
p2 = ax.bar(ind + width, female_means, width, color='blue')

ax.set_title('Scores by gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('attractive', 'sincere', 'intelligence',                     'funny', 'ambition', 'shared_interests'))

ax.legend((p1[0], p2[0]), ('male', 'female'))
ax.autoscale_view()

plt.savefig('./figs/2_1.png')
#plt.show()


# In[ ]:




