#!/usr/bin/env python
# coding: utf-8

# Cluster Analysis
# K-Means

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


plt.figure(figsize=(7,4))
iris = datasets.load_iris()

X = scale(iris.data)
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names
X[0:10,]


# Das Modell erstellen und ausführen

# In[33]:


clustering = KMeans(n_clusters=3, random_state=5)

clustering.fit(X)


# Die Ausgaben des Models zeichnen

# In[ ]:


iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']


# In[34]:


plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)

plt.title('Grundwahrheitsklassifikation')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
plt.title('K-Means-Klassifikation')


# In[36]:


relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Grundwahrheitsklassifikation')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[relabel], s=50)
plt.title('K-Means-Klassifikation')


# Bewerten der Clustering-Ergebnisse

# In[37]:


print(classification_report(y, relabel))


# In[ ]:





# In[ ]:




