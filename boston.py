#!/usr/bin/env python
# coding: utf-8
# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.rcParams['figure.figsize'] = (10,6)

# %% [markdown]
# # Boston Housing Data
#  
# Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. Concerns housing values in suburbs of Boston.
#  
#  - Number of Instances: 506
#  - Number of Attributes: 13 continuous attributes
#  
# ## Attribute Information:
#  
#  1. CRIM      per capita crime rate by town    
#  2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.                
#  3. INDUS     proportion of non-retail business acres per town    
#  4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)                 
#  5. NOX       nitric oxides concentration (parts per 10 million)   
#  6. RM        average number of rooms per dwelling
#  7. AGE       proportion of owner-occupied units built prior to 1940
#  8. DIS       weighted distances to five Boston employment centres
#  9. RAD       index of accessibility to radial highways
#  10. TAX      full-value property-tax rate per \$10,000
#  11. PTRATIO  pupil-teacher ratio by town
#  12. B $1000(B_k - 0.63)^2$ where $B_k$ is the proportion of blacks by town
#  13. LSTAT    % lower status of the population
#  14. MEDV     Median value of owner-occupied homes in \$ 1000's
#  
#  8. Missing Attribute Values:  None.

# %%
import numpy as np
import pandas as pd
df = pd.read_csv("data/boston.csv")
print(df.columns)
df.describe()


# %%


df.head()


# %%


df.CHAS = df['CHAS'].astype(np.bool)


# %%


df[df.CHAS].head()


# %%


df.aggregate({'AGE':['min', max, np.median, np.mean]})


# %%


df.corr()


# %%


sns.set_style("whitegrid")
sns.boxplot(data=df); 


# %%
from sklearn import preprocessing
df_scaled = preprocessing.scale(df)


# %%
sns.violinplot(data=df_scaled); 


# %%
df.plot( x="CRIM", y="MEDV", kind='scatter');


# %%
sns.regplot(x='CRIM',y='MEDV', data = df)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")


# %%
df.plot('RM','MEDV',kind='scatter');


# %%
df.plot('PTRATIO','MEDV',kind='scatter');


# %%
df.plot('ZN','MEDV',kind='scatter');


# %%
df.plot('INDUS','MEDV',kind='scatter', title="Relationship between INDUS and Price");


# %%
df.plot('NOX','MEDV',kind='scatter', title="Relationship between NOX and Price");


# %%
#Correlation Matrix

sns.set(style="white")

df_corr= df[:]
# Compute the correlation matrix
corr = df_corr.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, ax=ax)

# RAD and TAX are highly co-related.
#Price negatively corelated with LSTAT(Strong),PTRATIO(Strong),TAX(high), INDUS(High), CRIM(Highly) and 
#NOX highly corelated with RM.
#Also Price positively corelated with RM(High), ZN(High), CHAS(Medium), DIS(MEDIUM) & B(Medium)


# %%





# %%


sns.regplot(y="MEDV", x="RM", data=df, fit_reg = True);
plt.title("Relationship between average number of rooms per dwelling and PRICE")


# %%


sns.set()
plt.hist(df['CRIM'], ec='w');


# %%


plt.hist(df['RM'], ec='w');


# %%


plt.hist(df['PTRATIO'], ec='w');


# # References
# 
# - https://www.ritchieng.com/machine-learning-project-boston-home-prices/
# - https://github.com/chatkausik/Linear-Regression-using-Boston-Housing-data-set/blob/master/Mini_Project_Linear_Regression.ipynb
# - https://github.com/Tsmith5151/Boston-Housing-Prices/blob/master/boston_housing.ipynb

# %%


import numpy as np
import pandas as pd
import sklearn as sk


# %%


from sklearn.datasets import load_boston
boston = load_boston()


# %%


bos = pd.DataFrame(boston.data)


# %%


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# %%


X = bos.iloc[:,:-1] # drop PRICE column


# %%


lm.fit(X, bos.iloc[:,-1])


# %%


lm.predict(X)[0:5]


# %%


bos.iloc[:,-1][0:5]


# %%





# %%


X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(
    X, bos.PRICE, test_size=0.33, random_state = 5)
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)


# %%
