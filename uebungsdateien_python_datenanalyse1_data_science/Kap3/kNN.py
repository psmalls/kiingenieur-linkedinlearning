#!/usr/bin/env python
# coding: utf-8

# Cluster Analysis - kNN

# In[53]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Laden der Daten, auswählen, skalieren und teilen in Test- und Trainingsdatensätze
# 

# In[54]:


address = 'mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
X_prime = cars.iloc[:,[1,3,4,6]].values
y = cars.iloc[:,[9]].values


# In[55]:


X = preprocessing.scale(X_prime)


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=.33, random_state=17)


# Erstellen und trainieren des Modells mit Trainingsdaten
# 

# In[66]:


clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)
print(clf)


# Bewerten der Vorhersagen des Modells anhand des Testdatensatzes

# In[67]:


y_expect = y_test
y_pred = clf.predict(X_test)

print(metrics.classification_report(y_expect, y_pred))


# In[ ]:





# In[ ]:




