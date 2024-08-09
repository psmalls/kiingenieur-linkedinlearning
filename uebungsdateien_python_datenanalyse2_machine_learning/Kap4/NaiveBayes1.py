#!/usr/bin/env python
# coding: utf-8

# Naive Bayes

# In[ ]:


import numpy as np
import pandas as pd

import urllib.request

import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


# ## Naive Bayes
# ### Naive Bayes zum Schutz vor Spam

# In[41]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset[0])


# In[42]:


df = pd.DataFrame(dataset)
df.info()


# In[43]:


X = dataset[:,0:48]

y = dataset[:, -1]

dfX = pd.DataFrame(X)
dfX.info()
dfy = pd.DataFrame(y)
dfy.info()


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=17)


# In[45]:


dfX_train = pd.DataFrame(X_train)
dfX_train.info()


# In[46]:


dfX_test = pd.DataFrame(X_test)
dfX_test.info()


# In[47]:


dfy_train = pd.DataFrame(y_train)
dfy_train.info()


# In[48]:


dfy_test = pd.DataFrame(y_test)
dfy_test.info()


# In[ ]:




