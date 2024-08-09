#!/usr/bin/env python
# coding: utf-8

# Cluster Analysis - Hierarchische Methoden

# In[48]:


import numpy as np
import pandas as pd

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm


# In[49]:


address = 'mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

X = cars.iloc[:,[1,3,4,6]].values

y = cars.iloc[:,[9]].values


# Verwendung von scipy zum Erstellen des Dendrogramms

# In[50]:


Z = linkage(X, 'ward')


# In[51]:


dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)

plt.title('Verkürztes hierarchisches Clustering-Dendrogramm')
plt.xlabel('Clustergröße')
plt.ylabel('Entfernung')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()


# Hierarchische Cluster generieren

# In[52]:


k=2

Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
Hclustering.fit(X)

sm.accuracy_score(y, Hclustering.labels_)


# In[53]:


Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')
Hclustering.fit(X)

sm.accuracy_score(y, Hclustering.labels_)


# In[54]:


Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average')
Hclustering.fit(X)

sm.accuracy_score(y, Hclustering.labels_)


# In[55]:


Hclustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average')
Hclustering.fit(X)

sm.accuracy_score(y, Hclustering.labels_)


# In[ ]:




