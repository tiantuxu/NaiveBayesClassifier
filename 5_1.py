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


TRAINING_SET = 'trainingSet.csv'
TEST_SET = 'testSet.csv'
bin_N = 5

def nbc(t_frac):
    df = pd.read_csv(TRAINING_SET).sample(frac=t_frac, random_state=47)
    df_test = pd.read_csv(TEST_SET).sample(frac=t_frac, random_state=47)
    #df = df.astype('int64')
    #df_test = df_test.astype('int64')

    attr_list = list(df[df.columns.difference(['decision'])])
    dict_table ={}

    # Labels
    dict_labels = {}
    dict_labels['no'] = len(df[df['decision'] == 0])
    dict_labels['yes'] = len(df[df['decision'] == 1])
    dict_table['decision'] = dict_labels
    
    # Attributes in discrete_columns
    for attr in attr_list:
        dict_attr = {}
        attr_bin = max(int(df[attr].max()), int(df_test[attr].max()))
        
        dict_attr['no'] = [1 for i in range(attr_bin + 1)]
        dict_attr['yes'] = [1 for i in range(attr_bin + 1)]
        
        for i in range(attr_bin+1):
            dict_attr['no'][i] += len(df[(df[attr] == i) & (df['decision'] == 0)])
            dict_attr['yes'][i] += len(df[(df[attr] == i) & (df['decision'] == 1)])

        dict_table[attr] = dict_attr
        
    return dict_table


# ### Accuracy on training data

# In[3]:


# Print the accuracy
dict_table = nbc(1)
# Accuracy on training data
df = pd.read_csv(TRAINING_SET).sample(frac=1, random_state=47)
(row, col) = df.shape
attr_list = list(df[df.columns.difference(['decision'])])
#print df['decision'].value_counts()
neg_num = len(df[df['decision'] == 0])
pos_num = len(df[df['decision'] == 1])

print dict_table
row_index = df.index.tolist()

#print df
correct = 0
for i in row_index:
    pd_pos = 1.0 * dict_table['decision']['yes']/row
    pd_neg = 1.0 * dict_table['decision']['no']/row
    #print pd_pos, pd_neg
    for attr in attr_list:
        pd_pos *= 1.0 * dict_table[attr]['yes'][int(df[attr][i])]/pos_num
        pd_neg *= 1.0 * dict_table[attr]['no'][int(df[attr][i])]/neg_num
    
    res = np.argmax([1.0 * pd_neg, 1.0 * pd_pos])
    if res == df['decision'][i]:
        correct += 1
#print correct
training_accuracy = 1.0 * correct/row
print 'Training Accuracy:', '%.2f' % training_accuracy
#print 'Training Accuracy:', training_accuracy


# ### Accuracy on test data

# In[4]:


df_test = pd.read_csv(TEST_SET).sample(frac=1, random_state=47)
(row_test, col_test) = df_test.shape
#attr_list = list(df_test[df_test.columns.difference(['decision'])])

#neg_num_test = len(df_test[df_test['decision'] == 0])
#pos_num_test = len(df_test[df_test['decision'] == 1])

correct = 0

row_index_test = df_test.index.tolist()

for i in row_index_test:
    pd_pos = 1.0 * dict_table['decision']['yes']/row
    pd_neg = 1.0 * dict_table['decision']['no']/row
    #print pd_pos, pd_neg
    for attr in attr_list:
        pd_pos *= 1.0 * dict_table[attr]['yes'][int(df_test[attr][i])]/pos_num
        pd_neg *= 1.0 * dict_table[attr]['no'][int(df_test[attr][i])]/neg_num
    
    res = np.argmax([1000.0 * pd_neg, 1000.0 * pd_pos])
    if res == df_test['decision'][i]:
        correct += 1
#print correct tt
test_accuracy = 1.0 * correct/row_test
print 'Test Accuracy:', '%.2f' % test_accuracy
#print 'Test Accuracy:', test_accuracy


# In[ ]:




