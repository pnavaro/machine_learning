#!/usr/bin/env python
# coding: utf-8
# %%

# # Jeu de données Iris
# 
# ## Version pandas

# %%


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.rcParams['figure.figsize'] = (10,6)


# %%


import numpy as np
import pandas as pd
df = pd.read_csv("data/iris.csv")
print(df.columns)
df.describe() # species n'est pas dans la description (variable qualitative)


# %%


import pandas as pd


# %%


df.groupby('species').size()


# ## solutions

# %%


df.species.nunique() # nombre de classes (espèces) = 3
print(df.shape) # 150 individus (relativement peu)
print(df.groupby('species')['species'].count()) # solution 1 avec groupby()
print(df.species.value_counts()) # solution 2 (plus simple) avec value_counts()
# 50 individus par classe... pas si mal (et pas de déséquilibre)


# %%




plt.figure(figsize=(8,4)) 

df['y'] = df.species.astype('category') # conversion en variable catégorielle
df['y'] = df.species.replace({'setosa':1, 'versicolor':2, 'virginica':3}) # conversion en nombres
counts = df.species.value_counts()
plt.subplot(121)
counts.plot.pie(y='species')
plt.subplot(122)
counts.plot.bar(x=None, y='species', stacked=True); # stacked=True => barres devraient etre empilees


# %%


sns.set(style="darkgrid")
ax = sns.countplot(x="species", data=df) # directement sur df (et non counts) 


# %%


df.corr()


# Quels prédicteurs/features sont les plus importants ?
# (les plus susceptibles de discriminer les espèces d'IRIS)

# %%


sns.set_style("whitegrid")
sns.boxplot(data=df); 


# %%


sns.violinplot(data=df); 


# %%


df.columns[:-2]


# plus de variabilité dans "petal length", et très peu dans "sepal width"

# %%


sns.pairplot(df, hue='species',vars=df.columns[:-2],size=2);


# %%


sns.pairplot(df, kind="scatter", hue="species", vars=df.columns[:-2], markers=["o", "s", "D"], palette="Set2")


# ## Quelle(s) classe(s) sera(ont) facile(s) à distinguer ?
# 

# %%


sns.lmplot( x="sepal_length", y="sepal_width", 
           data=df, fit_reg=False, hue='species', legend=False)
plt.legend(loc='lower right'); 


# %%


sns.lmplot( x="petal_length", y="petal_width", data=df, 
           fit_reg=False, hue='species', legend=False)
plt.legend(loc='lower right'); 


# ## IRIS : conclusion
# Le jeu de données n'est pas très grand (150 individus et seulement 4 prédicteurs)
# les prédicteurs liées aux pétales ne sont pas distribuées normalement
# les variables sont corrélées (positivement), sauf "sepal width"
# "petal length" puis "petal width" semblent les prédicteurs les plus prometteurs
# 
# la classe Sertosa ne posera certainement aucun problème
# Toutes les analyses ne seront pas possibles (celles qui nécessitent la normalité, ou l'indépendance entre variables), mais on peut espérer des bons résultats...

# ## Version scikit-learn

# %%


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# %%


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# %%


scaler = StandardScaler()
scaler.fit(X_train)
X_train_tr = scaler.transform(X_train)
X_test_tr = scaler.transform(X_test)


# %%


X_test_tr.shape, y_train.shape


# %%


from sklearn.svm import SVC
model = SVC(kernel="linear", C=1.)
model.fit(X_train_tr, y_train) 

model.score(X_train_tr, y_train), model.score(X_test_tr, y_test)


# %%


from sklearn.metrics import accuracy_score


# %%


for gamma in [0.1,1.0, 10.]:
    model = SVC(kernel="rbf", C=1., gamma=gamma)
    model.fit(X_train_tr, y_train) 

    print(model.score(X_test_tr, y_test), end=',')
    print(accuracy_score(y_test, model.predict(X_test_tr)))


# %%


def plot_decision(X, y=None, model=None):
    if model is not None:        
        xx, yy = np.meshgrid(np.arange(X[:, 0].min() - .5, 
                                       X[:, 0].max() + .5, .01),
                             np.arange(X[:, 1].min() - .5, 
                                       X[:, 1].max() + .5, .01))
        zz_class = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, zz_class, alpha=.2)
    # Plot data
    if y is None:
        y = "k"
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, alpha=0.5, cmap='jet')
    # Set figure coordinate limits
    plt.xlim(X[:, 0].min() - .5, X[:, 0].max() + .5)
    plt.ylim(X[:, 1].min() - .5, X[:, 1].max() + .5)


# %%


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)


print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  


# %%


plot_decision(pca.transform(X), y)
sns.set_style("white")


# %%




plot_decision(pca.transform(X), y)
sns.set_style("white")


# ## Classification non supervisée

# %%


from sklearn.cluster import KMeans
from sklearn import datasets
iris = datasets.load_iris()
#Stocker les données en tant que DataFrame Pandas 
x = pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y = pd.DataFrame(iris.target)
y.columns=['Targets']
y.columns


# %%


#Cluster K-means
model=KMeans(n_clusters=3)
#adapter le modèle de données
model.fit(pca.transform(X))


# %%


print(model.labels_)


# %%


plt.figure()
plot_decision(pca.transform(X), y=None, model=model)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker="^", c="r", s= 50)


# %%





# %%
