#!/usr/bin/env python
# coding: utf-8

# Logistische Regression
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import spearmanr
import sklearn
from sklearn.preprocessing import scale 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[ ]:


sb.set_style('whitegrid')


# Logistische Regression - Datei mtcars.csv

# In[47]:


address = 'mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.head()


# In[48]:


cars_data = cars.iloc[:,[5,11]].values
cars_data_names = ['drat','carb']

y = cars.iloc[:,9].values


# Check Unabhängigkeit der Features

# In[49]:


sb.regplot(x='drat', y='carb', data=cars, scatter=True)


# In[50]:


drat = cars['drat']
carb = cars['carb']

spearmanr_coefficient, p_value =  spearmanr(drat, carb)
print ('Spearman Korrelationskoefficient %0.3f' % (spearmanr_coefficient))


# Check fehlende Werte (missing values)?

# In[51]:


cars.isnull().sum()


# Check Ziel (target) binär oder ordinal?

# In[52]:


sb.countplot(x='am', data=cars, palette='hls')


# Check Größe Dataset ausreichend?

# In[53]:


cars.info()


# Modell erstellen

# In[54]:


X = scale(cars_data)


# In[55]:


LogReg = LogisticRegression(solver='lbfgs')
LogReg.fit(X,y)
print (LogReg.score(X,y))


# In[56]:


y_pred = LogReg.predict(X)
print(classification_report(y, y_pred))


# In[ ]:




