#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split


# In[21]:


n = 100
np.random.seed(0)
X = np.random.randn(  n, 1)
X.shape


# In[22]:


eps = np.random.normal(0.0, 0.1,   n)


# In[23]:


y = np.sin(X[:,0]) + eps
plt.plot(X, y, '.');


# In[24]:


model = LinearRegression()


# In[25]:


model.fit(X, y)


# In[26]:


y_pred = model.predict(X)


# In[27]:


plt.plot(X, y_pred, 'r-', X, y, 'bo');


# In[28]:


n = 100
dim = 10
np.random.seed(0)
X = np.random.randn(n, dim)
X.shape


# In[29]:


eps = np.random.normal(0.0, 0.1,   n)
y = np.sin(X[:,0]) + eps
y.shape


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# In[31]:


X_train.shape, y_train.shape


# In[32]:


model = Lasso(alpha=0.2)
model.fit(X_train, y_train)


# In[33]:


y_pred = model.predict(X_test)
y_pred.shape


# In[34]:


plt.scatter(X_test[:,0], y_test,  color='b')
plt.plot(X_test[:,0], y_pred, color='r', linewidth=2)

