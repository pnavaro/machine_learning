#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10,6)


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer


# # "TD : `sklearn` & sélection de modèles"
# 
# Romain Tavenard - (Creative Commons CC BY-NC-SA)[http://creativecommons.org/licenses/by/4.0/]

# Dans cette séance, nous nous focaliserons sur la sélection de modèle pour la
# classification supervisée avec `sklearn`.
# 
# # Préparation des données
# 
# Nous allons travailler, pour ce TD, sur un jeu de données ultra classique en
# _machine learning_ : le jeu de données "Iris". Ce jeu de données est intégré
# dans `sklearn` pour être utilisé facilement.
# 
# 1. Chargez ce jeu de données à l'aide de la fonction [`load_iris`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) du module
# `sklearn.datasets`. Faites en sorte de stocker les prédicteurs dans une matrice
# `X` et les classes à prédire dans un vecteur `y`. Quelles sont les dimensions
# de `X` ?

# In[5]:


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X.shape


# 2. Découpez ce jeu de données en un jeu d'apprentissage et un jeu de test de
# mêmes tailles et faites en sorte que chacune de vos variables soient
# centrées-réduites.

# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_tr = scaler.transform(X_train)
X_test_tr = scaler.transform(X_test)


# # Le modèle `SVC` (_Support Vector Classifier_)
# 
# 3. Apprenez un modèle SVM linéaire (classe [`SVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) dans `sklearn`) pour votre
# problème.

# In[7]:


model = SVC(kernel="linear", C=1.)
model.fit(X_train_tr, y_train)
print(model.score(X_train_tr, y_train))
print(model.score(X_test_tr, y_test))


# 4. Évaluez ses performances sur votre jeu de test à l'aide de la fonction
# [`accuracy_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) du module `sklearn.metrics`.

# 5. Faites de même avec un modèle SVM à noyau gaussien. Faites varier la valeur
# de l'hyperparamètre lié à ce noyau et observez l'évolution du taux de bonnes
# classifications.

# # Validation croisée
# 
# Il existe dans `sklearn` de nombreux itérateurs permettant de faire de la
# validation croisée, qui sont listés sur
# [cette page](http://scikit-learn.org/stable/modules/classes.html#splitter-classes).
# 
# 6. Définissez un objet `cv` de la classe `KFold`. Exécutez le code suivant :

# In[11]:


from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=True)
for train, valid in cv.split(X_train, y_train):
    print(train, valid, end='\n==\n')


# Qu'est-ce qui est affiché ?
# 
# 7. Faites de même avec des objets des classes `StratifiedKFold` et `LeaveOneOut`
# et vérifiez que, même en ne mélangeant pas les données (c'est-à-dire sans
# spécifier `shuffle=True`), les découpages obtenus sont différents.

# In[12]:


from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)
for train, valid in cv.split(X_train, y_train):
    print(train, valid, end='\n==\n')


# In[13]:


from sklearn.model_selection import LeaveOneOut

cv = LeaveOneOut()
for train, valid in cv.split(X_train, y_train):
    print(train, valid, end='\n==\n')


# # Sélection de modèle
# 
# En pratique, vous n'utiliserez pas vous même ces appels aux méthodes `split()`
# des itérateurs de validation croisée, car il existe dans `sklearn` un outil
# très pratique pour vous aider lors de votre étape de sélection de modèle.
# 
# Cet outil s'appelle
# [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
# Là encore, il s'agit d'une classe `sklearn`, et vous l'utiliserez quasiment de
# la même manière qu'un classifieur, à la nuance près des paramètres que vous
# passerez lors de la construction d'une nouvelle instance.
# Nous verrons dans ce TD trois de ces paramètres :
# 
# * `estimator` est un classifieur (créé mais pas encore appris sur des données) ;
# * `param_grid` est une grille d'hyper-paramètres à tester pour ce classifieur ;
# * `cv` est un itérateur de validation croisée, tel que l'un de ceux définis à la
# section précédente.
# 
# Le paramètre `param_grid` est un dictionnaire (ou une liste de dictionnaire,
# comme on le verra plus tard) dont les clés sont les noms des hyper-paramètres à
# faire varier et les valeurs associées sont des listes de valeurs à
# tester[^1].
# 
# 8. Reprenez le cas d'un classifieur SVM linéaire et faites varier
# l'hyper-paramètre `C` entre 1 et 10 (en prenant 5 valeurs espacées
#     régulièrement).

# In[14]:


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, KFold
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1,3,5,7,10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters , cv=KFold(n_splits=10))
clf.fit(iris.data, iris.target)


# 9. Affichez les paramètres du modèle choisi par la validation croisée. Évaluez
# les performances de ce modèle sur votre jeu de test.

# 10. Parfois, certains hyper-paramètres sont liés entre eux. Dans le cas du SVM
# par exemple, le choix d'un noyau implique de régler certains hyper-paramètres
# spécifiques (_ex._ : le paramètre `gamma` du noyau Gaussien). Dans ce cas, on
# peut définir `param_grid` comme une liste de dictionnaires, chaque dictionnaire
# correspondant à un cas de figure. Utilisez cette possibilité pour choisir le
# meilleur modèle pour votre problème entre un SVM linéaire et un SVM à noyau
# Gaussien (pour les deux, on fera varier `C` entre 1 et 10, et pour le noyau Gaussien, on fera de plus varier `gamma` entre 10<sup>-2</sup> et 100 sur une échelle logarithmique).

# In[15]:


parameters = {'kernel':('linear', 'rbf'), 'C': np.linspace(1,10,5), 'gamma':np.logspace(-2,2,5)}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, return_train_score=True, cv=KFold(n_splits=10))
clf.fit(iris.data, iris.target)
clf.best_params_, clf.best_score_        


# In[16]:


clf.cv_results_


# 11. Étendez cette approche à un autre classifieur supervisé de votre choix et
# comparez ses performances à celles du meilleur modèle SVM trouvé jusqu'alors.

# In[17]:


import pandas as pd
results = pd.DataFrame(clf.cv_results_)
results['param_C']


# In[18]:


results = results.loc[:,['param_kernel','param_C','mean_train_score','mean_test_score']]
results.groupby(['param_kernel','param_C']).max()


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
parameters = {'kernel':('linear', 'rbf'), 'C': np.linspace(1,10,5), 'gamma':np.logspace(-2,2,5)}
estimator = KNeighborsClassifier()
clf = GridSearchCV(estimator=estimator, 
                   param_grid={"n_neighbors":[1,5,10]}, 
                   return_train_score=True, 
                   cv=KFold(n_splits=10, shuffle=True))
clf.fit(iris.data, iris.target)
clf.best_params_, clf.best_score_  


# In[20]:


clf.score(X_test, y_test)


# In[ ]:





# # La notion de _Pipeline_
# 
# Bien souvent, pour mettre en oeuvre une solution de _machine learning_, vous
# allez passer par plusieurs étapes successives de transformation de vos données
# avant de les fournir à votre classifieur. Il est possible que ces
# pré-traitements aient, eux aussi, des hyper-paramètres à régler. Pour pouvoir
# sereinement prendre en compte toutes les configurations possibles, `sklearn`
# définit la notion de
# [`Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
# 
# 12. Modifiez vos données d'origine pour mettre des `numpy.nan` à toutes les
# valeurs plus grandes que 2 (en valeur absolue).

# 13. Créez un _pipeline_ qui soit constitué des 3 étapes suivantes :
# a. une imputation des valeurs manquantes ;
# b. une standardisation des données ;
# c. une classification supervisée par un classifieur de votre choix.

# 14. Mettez en place une validation croisée qui permette de choisir si
# l'imputation doit se faire par valeurs médianes ou moyennes et qui détermine
# également un ou plusieurs hyper-paramètres du classifieur que vous avez choisi.

# 15. Chargez maintenant de nouvelles données à l'aide de la fonction
# [`load_digits`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
# du module `sklearn.datasets`.
# Imaginez un _pipeline_ qui consiste à effectuer tout d'abord une ACP sur ces
# données puis une régression logistique dans l'espace de l'ACP.
# Mettez en place une validation croisée pour choisir de manière optimale les
# paramètres de ces deux étapes de votre _pipeline_.
# Comparez votre solution à celle disponible
# [là](http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html).
# 
# [^1]: Pour générer ces listes, on pourra avoir recours aux fonctions [`numpy.linspace`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linspace.html) et [`numpy.logspace`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.logspace.html).

# In[21]:


pd.__version__


# In[23]:


from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

X[X>2] = np.nan
imp = Imputer()
scaler = StandardScaler()
clf = SVC(kernel="linear")
model_pipeline = Pipeline(steps=[("imp",imp), ("scaler", scaler), ("svc", clf)])
gs = GridSearchCV(estimator=model_pipeline,
                  param_grid={"imp__strategy": ["mean","median"],
                             "svc__C": np.linspace(1,10,5)},
                  cv=KFold(n_splits=10, shuffle=True))
gs.fit(X,y)
gs.best_params_


# In[ ]:



# Cross validation

# cv = KFold(n_splits=10, shuffle=True)
# cv = StratifiedKFold(n_splits=10)
cv = LeaveOneOut()
for train, valid in cv.split(X_train, y_train):
    print(train, valid)
    print("\n===\n")

# Grid Search

# clf = GridSearchCV(estimator=SVC(),
#                    param_grid=[{"kernel": ["linear"], "C": numpy.linspace(1, 10, 5)},
#                                {"kernel": ["rbf"], "C": numpy.linspace(1, 10, 5),
#                                 "gamma": numpy.logspace(-2, 2, 5)}],
#                    cv=KFold(n_splits=10, shuffle=True))
clf = GridSearchCV(estimator=KNeighborsClassifier(),
                   param_grid={"n_neighbors": [10, 5, 1]},
                   cv=KFold(n_splits=10, shuffle=True))
clf.fit(X_train, y_train)
print(clf.best_params_, clf.best_score_)
print(clf.score(X_test, y_test))

# Pipeline

X[X>2] = numpy.nan

imp = Imputer()
scaler = StandardScaler()
clf = SVC(kernel="linear")

model_pipeline = Pipeline(steps=[("imp", imp), ("scaler", scaler), ("svc", clf)])
gs = GridSearchCV(estimator=model_pipeline,
                  param_grid={"imp__strategy": ["mean", "median"],
                              "svc__C": numpy.linspace(1, 10, 5)},
                  cv=KFold(n_splits=10, shuffle=True))

gs.fit(X, y)

print(gs.best_params_)


# In[ ]:




