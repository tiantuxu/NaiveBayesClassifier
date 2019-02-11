#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd


# In[2]:


#input_filename = argv[1]
#output_filename = argv[2]
sys.argv[1] = 'dating-full.csv'
sys.argv[2] = 'dating.csv'
# Load csv
d = pd.read_csv(sys.argv[1])


# ## Remove Quotes

# In[3]:


quote = 0
(row, col) = d.shape
#print row
#print col
for i in range(row):
    if d['race'][i].startswith("'") and d['race'][i].endswith("'"):
        quote += 1
    if d['race_o'][i].startswith("'") and d['race_o'][i].endswith("'"):
        quote += 1
    if d['field'][i].startswith("'") and d['field'][i].endswith("'"):
        quote += 1

d['race'] = d['race'].str.replace("'","")
d['race_o'] = d['race_o'].str.replace("'","")
d['field'] = d['field'].str.replace("'","")

print 'Quotes removed from', quote, 'cells.'


# ## Change to lower case

# In[4]:


case = 0
for i in range(row):
    if any(letter.isupper() for letter in str(d['field'][i])):
        case += 1

d['field'] = d['field'].str.lower()
print 'Standardized', case, 'cells to lower case.'


# ## Categorical to numeric

# In[5]:


# gender
gender = []
for i in d['gender']:
    if i not in gender:
        gender.append(i)
gender.sort()
d['gender'] = d['gender'].astype('category')
d['gender'] = d['gender'].cat.codes

# race
race = []
#print d['race']
for i in d['race']:
    if i not in race:
        race.append(i)
race.sort()
d['race'] = d['race'].astype('category')
d['race'] = d['race'].cat.codes
#print d['race']

# race_o
race_o = []
for i in d['race_o']:
    if i not in race_o:
        race_o.append(i)
race_o.sort()
d['race_o'] = d['race_o'].astype('category')
d['race_o'] = d['race_o'].cat.codes

# field
field = []
for i in d['field']:
    if i not in field:
        field.append(i)
field.sort()
d['field'] = d['field'].astype('category')
d['field'] = d['field'].cat.codes

print 'Value assigned for male in column gender:', str(gender.index('male')) + '.'
print 'Value assigned for European/Caucasian-American in column race:', str(race.index('European/Caucasian-American')) + '.'
print 'Value assigned for Latino/Hispanic American in column race_o:', str(race_o.index('Latino/Hispanic American')) + '.'
print 'Value assigned for law in column field:', str(field.index('law')) + '.'


# In[6]:


preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',  'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']

for i in range(row):
    participant_sum = 0
    partner_sum = 0

    for pref in preference_scores_of_participant:
        participant_sum += d[pref][i]
        
    for pref in preference_scores_of_partner:
        partner_sum += d[pref][i]
    
    # update the preference scores of participant
    for pref in preference_scores_of_participant:
        d.loc[i, pref] = d[pref][i]/participant_sum
        
    # update the preference scores of partner
    for pref in preference_scores_of_partner:
        d.loc[i, pref] = d[pref][i]/partner_sum

    
for pref in preference_scores_of_participant:
    print 'Mean of', pref + str(':'), str('%.2f' % d[pref].mean()) + '.'
    
for pref in preference_scores_of_partner:
    print 'Mean of', pref + str(':'), str('%.2f' % d[pref].mean()) + '.'


# ## Save csv

# In[7]:


d.to_csv(sys.argv[2], index=False)

