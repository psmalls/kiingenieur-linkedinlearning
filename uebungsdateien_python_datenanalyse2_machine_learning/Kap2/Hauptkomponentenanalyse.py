#!/usr/bin/env python
# coding: utf-8

# PCA

# In[65]:


import numpy as np
import pandas as pd
import seaborn as sb
import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets


# ### PCA in Verbindung mit dem iris Dataset

# In[66]:


iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names
X[0:10,]


# In[71]:


pca = decomposition.PCA()
iris_pca = pca.fit_transform(X)

pca.explained_variance_ratio_


# In[68]:


pca.explained_variance_ratio_.sum()


# In[69]:


comps = pd.DataFrame(pca.components_, columns=variable_names)
comps


# In[72]:


sb.heatmap(comps)


# In[ ]:





# In[ ]:




