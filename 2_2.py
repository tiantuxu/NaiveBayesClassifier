#!/usr/bin/env python
# coding: utf-8

# In[58]:


import sys, os
import pandas as pd
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# In[59]:


# Read the data
df = pd.read_csv('dating.csv')


# In[60]:


rating_of_partner_from_participant = ['attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner']


# In[61]:


TOTAL_SAMPLE = len(df)
rating_of_partner_from_participant_stat = {}
rating_of_partner_from_participant_decision = {}

for pref in rating_of_partner_from_participant:
    rating_of_partner_from_participant_stat[pref] = df[pref].value_counts()
    rating_of_partner_from_participant_decision[pref] = pd.crosstab(index=df[pref], columns=df['decision'])[1]

#print rating_of_partner_from_participant_stat
#print rating_of_partner_from_participant_decision


# In[65]:


# Four polar axes
#f, axarr = plt.subplots(2, 3, subplot_kw=dict(projection='polar'))
plt.figure(figsize=(100,60))
f, axarr = plt.subplots(3, 2, figsize=(10,10))
for i in range(len(rating_of_partner_from_participant)):
    X = []
    Y = []
    for v in df[rating_of_partner_from_participant[i]].unique():
        Y.append(1.0 * rating_of_partner_from_participant_decision[rating_of_partner_from_participant[i]][v] /                  rating_of_partner_from_participant_stat[rating_of_partner_from_participant[i]][v])
        X.append(v)
        axarr[i/2, i%2].scatter(X, Y)
        axarr[i/2, i%2].set_title(rating_of_partner_from_participant[i])
        axarr[i/2, i%2].set_xlabel('values')
        axarr[i/2, i%2].set_ylabel('success rate')
        #axarr[i/2, i%2].set_figheight(15)
        #axarr[i/2, i%2].set_figwidth(15)

# Fine-tune figure; make subplots farther from each other.
f.subplots_adjust(hspace=1, wspace = 0.2)
plt.savefig('./figs/2_2.png')
#plt.show()

