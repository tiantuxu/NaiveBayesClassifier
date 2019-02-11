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


# Read the data
df = pd.read_csv('dating-binned.csv')


# In[3]:


# Get the 
df_test = df.sample(frac=0.2, random_state=47)
df_test.to_csv('testSet.csv', index=False)
#print df_test.index
# Subtract 
df_train = df[~df.index.isin(df_test.index)]
df_train.to_csv('trainingSet.csv', index=False)
#print df_train.index

