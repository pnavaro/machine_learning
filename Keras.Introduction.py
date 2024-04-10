# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TD : `keras` & perceptron multi-couches
#
# Romain Tavenard
# Creative Commons CC BY-NC-SA

# Dans cette séance, nous nous focaliserons sur la création et l'étude de modèles
# de type perceptron multi-couches à l'aide de la librairie `keras`.
#
# Pour cela, vous utiliserez la classe de modèles `Sequential()` de `keras`.
# Voici ci-dessous un exemple de définition d'un tel modèle :

# %env KERAS_BACKEND=theano

# +
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist, boston_housing
from keras.utils import to_categorical
import numpy as np



# +
#1. définir les couches et les ajouter l'une après l'autre au modèle
premiere_couche = Dense(units=12, activation="relu", input_dim=24)
couche_cachee = Dense(units=12, activation="sigmoid")
couche_sortie = Dense(units=3, activation="linear")

model = Sequential()
model.add(premiere_couche)
model.add(couche_cachee)
model.add(couche_sortie)
# -

# 2. Spécifier l'algo d'optimisation et la fonction de risque à optimiser
#
# Fonctions de risque classiques :
#  * "mse" en régression,
#  * "categorical_crossentropy" en classification multi-classes
#  * "binary_crossentropy" en classification binaire
#  
#  On peut y ajouter des métriques supplémentaires (ici taux de bonne
#  classifications)

model.compile(optimizer="sgd", loss="mse", metrics=["accuracy"])

# 3. Lancer l'apprentissage

# +
#model.fit(X_train, y_train, verbose=2, epochs=10, batch_size=200)
# -

# # Préparation des données
#
# Pour ce TD, nous vous proposons d'utiliser les fonctions suivantes pour préparer
# vos données à l'utilisation dans `keras` :

from sklearn.preprocessing import MinMaxScaler
from keras.datasets import mnist, boston_housing
from keras.utils import to_categorical
import numpy as np


# +
def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test



# +
def prepare_boston():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    x_train = scaler_x.transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train[:,np.newaxis])
    y_train = scaler_y.transform(y_train[:,np.newaxis])
    y_test = scaler_y.transform(y_test[:,np.newaxis])
    return x_train, x_test, y_train, y_test



# -

x_train, x_test, y_train, y_test = prepare_boston()
x_train.shape, y_train.shape

# +

x_train.shape, y_train.shape
# -

# 1. Observez le code des fonctions `prepare_mnist` et `prepare_boston`.
# Que font ces fonctions ? Quelles sont les dimensions des matrices / vecteurs à
# la sortie ? S'agit-il de problèmes de classification ou de régression ?

# # Premiers réseaux de neurone
#
# 2. Chargez le jeu de données MNIST et apprenez un premier modèle sans couche
# cachée avec une fonction d'activation raisonnable pour les neurones de la couche
# de sortie. Pour cette question comme pour les suivantes, limitez vous à un
# nombre d'itérations de l'ordre de 10 : ce n'est absolument pas réaliste, mais
# cela vous évitera de perdre trop de temps à scruter l'apprentissage de vos
# modèles.

# regression logicstique
x_train, x_test, y_train, y_test = prepare_mnist()
x_train.shape, y_train.shape

# +
input_layer = Dense(units=y_train.shape[1], 
                    activation="softmax", 
                    input_dim=x_train.shape[1])

model = Sequential()
model.add(input_layer)
model.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=256, verbose=2)
# -

# 3. Comparez les performances de ce premier modèle è celle de modèles avec
# respectivement 1, 2 et 3 couches cachées de 128 neurones chacune. Vous
# utiliserez la fonction ReLU (`"relu"`) comme fonction d'activation pour les
# neurones des couches cachées.

# +
input_layer = Dense(units=128, 
                    activation="relu", 
                    input_dim=x_train.shape[1])

second_layer = Dense(units=128, 
                    activation="relu", 
                    input_dim=x_train.shape[1])


third_layer = Dense(units=y_train.shape[1], 
                    activation="softmax")

model = Sequential()
model.add(input_layer)
model.add(second_layer)
model.add(third_layer)

model.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=256, verbose=2)
# -

# 4. On peut obtenir le nombre de paramètres d'un modèle à l'aide de la
# méthode `count_params()`. Comptez ainsi le nombre de paramètres du modèle à
# 3 couches cachées et définissez un modèle à une seule couche cachée ayant un
# nombre comparable de paramètres. Parmi ces deux modèles, lequel semble le plus
# performant ?

# +
input_layer = Dense(units=128, 
                    activation="relu", 
                    input_dim=x_train.shape[1])

second_layer = Dense(units=128, 
                    activation="relu", 
                    input_dim=x_train.shape[1])


third_layer = Dense(units=y_train.shape[1], 
                    activation="softmax")

model = Sequential()
model.add(input_layer)
model.add(second_layer)
model.add(third_layer)

model.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

model.fit(x_train, y_train, 
          epochs=10, batch_size=256, 
          verbose=2, validation_split=0.1)

y_pred = model.predict(x_train)

confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1))
# -

model.count_params()

# # Utilisation d'un jeu de validation
#
# Bien entendu, les observations faites plus haut ne sont pas suffisantes,
# notamment parce qu'elles ne permettent pas de se rendre compte de l'ampleur du
# phénomène de sur-apprentissage.
#
# Pour y remédier, `keras` permet de fixer, lors de l'appel à la méthode `fit()`,
# une fraction du jeu d'apprentissage à utiliser pour la validation.
# Jetez un oeil
# [ici](https://keras.io/getting-started/faq/#how-is-the-validation-split-computed)
# pour comprendre comment les exemples de validation sont
# choisis.

# 5. Répétez les comparaisons de modèles ci-dessus en vous focalisant sur le taux
# de bonnes classifications obtenu sur le jeu de validation (vous prendrez 30\%
#     du jeu d'apprentissage pour votre validation).

# # Régularisation et _Drop-Out_
#
# 6. Appliquez une régularisation de type $L_1$ à chacune des couches de votre
# réseau. L'aide disponible [ici](https://keras.io/regularizers/) devrait
# vous aider.

# 7. Au lieu de la régularisation $L_1$, choisissez de mettre en place une
# stratégie de [_Drop-Out_](https://keras.io/layers/core/#dropout) pour aider à la
# régularisation de votre réseau.
# Vous éteindrez à chaque étape 10\% des poids de votre réseau.

# # Algorithme d'optimisation et vitesse de convergence
#
# 8. Modifiez la méthode d'optimisation choisie. Vous pourrez notamment essayer
# les algorithmes `"rmsprop"` et `"adam"`, reconnus pour leurs performances.

# 9. En utilisant l'aide fournie [ici](https://keras.io/optimizers/), faites
# varier le paramètre `lr` (_learning rate_) à l'extrême pour observer :
#
# * l'instabilité des performances lorsque celui-ci est trop grand ;
# * la lenteur de la convergence lorsque celui-ci est trop petit.

# # Modèles `keras` dans `sklearn`
#
# Il est possible de transformer vos modèles `keras` (en tout cas, ceux qui sont
#     de type `Sequential`) en modèles `sklearn`. Cela a notamment pour avantage
# de vous permettre d'utiliser les fonctionnalités de sélection de modèles vues
# lors du TD précédent.
#
# Pour cela, vous devrez utiliser au choix l'une des classes `KerasClassifier` ou
# `KerasRegressor` (selon le problème de _machine learning_ auquel vous êtes
#     confronté) du module `keras.wrappers.scikit-learn`.
#
# Le principe de fonctionnement de ces deux classes est le même :

# +
# %%time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist, boston_housing
from keras.utils import to_categorical
import numpy


def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test


def prepare_boston():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    x_train = scaler_x.transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train[:, numpy.newaxis])
    y_train = scaler_y.transform(y_train[:, numpy.newaxis])
    y_test = scaler_y.transform(y_test[:, numpy.newaxis])
    return x_train, x_test, y_train, y_test


def multi_layer_perceptron(input_dim, n_classes, activation="relu", optimizer="sgd"):
    premiere_couche = Dense(units=128,
                            activation=activation,
                            input_dim=input_dim)
    deuxieme_couche = Dense(units=128,
                            activation=activation)
    troisieme_couche = Dense(units=n_classes,
                            activation="softmax")

    model = Sequential()
    model.add(premiere_couche)
    model.add(deuxieme_couche)
    model.add(troisieme_couche)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model






X_train, X_test, y_train, y_test = prepare_mnist()

indices = numpy.random.permutation(X_train.shape[0])
X_train = X_train[indices]
y_train = y_train[indices]

# premiere_couche = Dense(units=128,
#                         activation="relu",
#                         input_dim=X_train.shape[1])
# deuxieme_couche = Dense(units=128,
#                         activation="relu")
# troisieme_couche = Dense(units=y_train.shape[1],
#                         activation="softmax")
#
# model = Sequential()
# model.add(premiere_couche)
# model.add(deuxieme_couche)
# model.add(troisieme_couche)
#
# model.compile(optimizer="adam",
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=2, validation_split=.1)
#
# y_pred = model.predict(X_train)
# print(confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1)))
#
# print(model.count_params())

clf = KerasClassifier(build_fn=multi_layer_perceptron,
                      input_dim=784,
                      n_classes=10,
                      activation="relu",
                      epochs=10,
                      batch_size=256,
                      verbose=2)
# clf.fit(X_train, y_train)

model_cv = GridSearchCV(estimator=clf,
                        param_grid={"optimizer": ["sgd", "adam", "rmsprop"]},
                        cv=KFold(n_splits=3))

model_cv.fit(X_train, y_train)
print(model_cv.best_params_, model_cv.best_score_)

# -

clf = KerasClassifier(build_fn=multi_layer_perceptron,
                      param_grid={"optimizer":}, 
                      param2="sgd", ...)
clf.fit(X, y)
clf.predict(X_test)

# Une fois construit, l'objet `clf` s'utilise donc exactement comme un classifieur
# `sklearn`.
# L'attribut `build_fn` prend le nom d'une fonction qui retourne un modèle
# `keras`. Les autres paramètres passés lors de la construction du classifieur
# peuvent être :
#
# * des paramètres de votre fonction `ma_fonction` ;
# * des paramètres passés au modèle lors de son apprentissage (appel à la
#     méthode `fit()`).

# 10. Créez un réseau à deux couches cachées transformé en objet `sklearn` en
# spécifiant, lors de sa construction, le nombre d'itérations et la taille des
# _batchs_ de votre descente de gradient par _mini-batchs_. Vous pourrez
# utiliser la méthode `score()` des objets `sklearn` pour évaluer ce modèle.

# 11. Utilisez les outils de validation croisée de `sklearn` pour choisir entre
# les algorithmes d'optimisation `"rmsprop"` et `"sgd"`.

# # La notion de `Callback`
#
# Les _Callbacks_ sont des outils qui, dans `keras`, permettent d'avoir un oeil
# sur ce qui se passe lors de l'apprentissage et, éventuellement, d'agir sur cet
# apprentissage.
#
# Le premier _callback_ auquel vous pouvez accéder simplement est retourné
# lors de l'appel à la méthode `fit()` (sur un objet `keras`, pas `sklearn`). Ce
# _callback_ est un objet qui possède un attribut `history`. Cet attribut est un
# dictionnaire dont les clés sont les métriques suivies lors de l'apprentissage.
# À chacune de ces clés est associé un vecteur indiquant comment la quantité en
# question a évolué au fil des itérations.

# 12. Tracez les courbes d'évolution du taux de bonnes classifications sur les
# jeux d'entrainement et de validation.
#
# La mise en place d'autres _callbacks_ doit être explicite. Elle se fait en
# passant une liste de _callbacks_ lors de l'appel à la méthode `fit()`.
# Lorsque l'apprentissage prend beaucoup de temps, la méthode précédente n'est pas
# satisfaisante car il est nécessaire d'attendre la fin du processus
# d'apprentissage avant de visualiser ces courbes. Dans ce cas, le _callback_
# [`TensorBoard`](https://keras.io/callbacks/#tensorboard) peut s'avérer très
# pratique.

# 13. Visualisez dans une page TensorBoard l'évolution des métriques `"loss"`
# et `"accuracy"` lors de l'apprentissage d'un modèle.
#
# De même, lorsque l'apprentissage est long, il peut s'avérer souhaitable
# d'enregistrer des modèles intermédiaires, dans le cas où un plantage arriverait
# par exemple. Cela peut se faire à l'aide du _callback_
# [`ModelCheckpoint`](https://keras.io/callbacks/#modelcheckpoint).

# 14. Mettez en place un enregistrement des modèles intermédiaires toutes les 2
# itérations, en n'enregistrant un modèle que si le risque calculé sur le jeu de
# validation est plus faible que celui de tous les autres modèles enregistrés
# aux itérations précédentes.

# 15. Mettez en oeuvre une politique d'arrêt précoce de l'apprentissage au cas où
# le risque calculé sur le jeu de validation n'a pas diminué depuis au moins 5
# itérations (en utilisant le _callback_
# [`EarlyStopping`](https://keras.io/callbacks/#earlystopping)).

# # Exercice de synthèse
#
# 16. Mettez en place une validation croisée pour choisir la structure (nombre de
#     couches, nombre de neurones par couche) et l'algorithme d'optimisation
#     idoines pour le problème lié au jeu de données _Boston Housing_ (pour lequel
#         une fonction de préparation des données est fournie dans le module
#         `dataset_utils`).

# +







X_train, X_test, y_train, y_test = prepare_mnist()

indices = numpy.random.permutation(X_train.shape[0])
X_train = X_train[indices]
y_train = y_train[indices]

# premiere_couche = Dense(units=128,
#                         activation="relu",
#                         input_dim=X_train.shape[1])
# deuxieme_couche = Dense(units=128,
#                         activation="relu")
# troisieme_couche = Dense(units=y_train.shape[1],
#                         activation="softmax")
#
# model = Sequential()
# model.add(premiere_couche)
# model.add(deuxieme_couche)
# model.add(troisieme_couche)
#
# model.compile(optimizer="adam",
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=2, validation_split=.1)
#
# y_pred = model.predict(X_train)
# print(confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1)))
#
# print(model.count_params())

clf = KerasClassifier(build_fn=multi_layer_perceptron,
                      input_dim=784,
                      n_classes=10,
                      activation="relu",
                      epochs=10,
                      batch_size=256,
                      verbose=2)
# clf.fit(X_train, y_train)

model_cv = GridSearchCV(estimator=clf,
                        param_grid={"optimizer": ["sgd", "adam", "rmsprop"]},
                        cv=KFold(n_splits=3))

model_cv.fit(X_train, y_train)
print(model_cv.best_params_, model_cv.best_score_)
# -


