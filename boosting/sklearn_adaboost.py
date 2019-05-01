#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


# 
# # Adaboost Classifier from scikit-learn

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier


# In[3]:


plt.rcParams["figure.figsize"] = (10,6)


# In[4]:


h = .02  # step size in the mesh

name = "AdaBoost"

clf = AdaBoostClassifier(n_estimators = 50)


# In[5]:


with open("data_banknote_authentication.txt") as f:
    data = f.read()

n = len(data)
X = np.zeros((n,4))
y = np.zeros(n)
for i,line in enumerate(data.splitlines()):
    line = line.split(",")
    X[i,:], y[i] = line[:4], line[4]

y -= (y == 0)


# In[13]:


X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=.4, random_state=42)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(f" score = {score} ")

plt.figure(1)
plt.rcParams["figure.figsize"] = [20,10]
fig,axs=plt.subplots(1,2)
axs[0].scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.viridis,marker='o')
axs[0].set_title('Données d entraînement',fontsize=14)
axs[1].scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.viridis,marker='o')
axs[1].set_title('Données classifiées par '+name,fontsize=14)


# In[ ]:




