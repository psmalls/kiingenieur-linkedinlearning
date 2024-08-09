#!/usr/bin/env python
# coding: utf-8

# Lineare Regression

# In[106]:


import numpy as np
import pandas as pd
import seaborn as sb
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale


# In[107]:


sb.set_style('whitegrid')


# #### (Multiple) linear regression on the enrollment data

# In[114]:


address = 'enrollment_forecast.csv'
enroll = pd.read_csv(address)
enroll.columns = ['year','roll','unem', 'hgrad', 'inc']
enroll.head()


# In[115]:


sb.pairplot(enroll)


# In[116]:


print (enroll.corr())


# In[117]:


enroll_data = enroll.iloc[:,[2,3]].values
enroll_target = enroll.iloc[:,1].values
enroll_data_names = ['unem', 'hgrad']

X, y = scale(enroll_data), enroll_target
print(X)
print("*"*50)
print(y)


# Missing values

# In[118]:


missing_values = X==np.NAN
X[missing_values == True]


# In[119]:


LinReg = LinearRegression(normalize=True)

LinReg.fit(X,y)

print (LinReg.score(X,y))


# In[ ]:




