#!/usr/bin/env python
# coding: utf-8

# Faktoranalyse
# 

# In[8]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import FactorAnalysis
from sklearn import datasets


# In[9]:


iris =  datasets.load_iris()
X = iris.data
variable_names = iris.feature_names
X[0:10,]


# Faktoranalyse auf dem iris-Dataset

# In[10]:


factor = FactorAnalysis().fit(X)
pd.DataFrame(factor.components_, columns=variable_names)


# In[ ]:




