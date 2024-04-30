# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: valerie
#     language: python
#     name: valerie
# ---

import sys
sys.executable

# ## Generate observations from a logit model

# +
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (9,6)

n = 1000

def logit(x, β):
    l = x.dot(β)
    return 1/(1+np.exp(-l))

rng = np.random.RandomState(1234)

β = [1, -2, 3, 0, 6, -1, 0]
rng = np.random.RandomState(1234)
X = rng.randn(n*len(β)).reshape(n, len(β))
p = logit(X, β)
y = np.random.binomial(1., p)
model = sm.Logit(y, X).fit()
print(model.summary())


# -

# ### Function that returns the maximum likehood estimator

def mle(X, y):
    model = sm.Logit(y, X).fit(disp=False)
    return model.llf


mle(X, y)


# +
def cv(X, y):
    nobs = np.size(X,0)
    score = 0
    indices = np.arange(nobs)
    for i in range(nobs):
        sapp = indices != i
        sval = i
        model = sm.Logit(y[sapp], X[sapp,:]).fit(disp=False)
        ypred = np.round(model.predict(X[sval, :]))
        score += y[sval] == ypred
    return (score / nobs)
        

cv(X, y)   

# +
import statsmodels.api as sm
import pandas as pd

def forward_regression(X, y, threshold_in = 0.01):
    included = []
    nvars = np.size(X, 1)
    while True:
        changed=False
        excluded = list(set(range(nvars))-set(included))
        new_pval = np.full(nvars, np.inf)
        for new_column in excluded:
            X̃ = sm.add_constant(X[:,included+[new_column]])
            model = sm.Logit(y, X̃).fit(disp=False)
            new_pval[new_column] = model.pvalues[-1]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = np.argmin(new_pval)
            included.append(best_feature)
            print(included)
            changed=True
        if not changed:
            return included

    


forward_regression(X, y, 0.05)


# +
def backward_regression(X, y, threshold_out = 0.01):
    included=list(range(np.size(X,1))) # all variables
    while True:
        changed=False
        X̃ = sm.add_constant(X[:,included])
        model = sm.Logit(y, X̃).fit(disp=False)
        # use all coefs except intercept
        pvalues = model.pvalues[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        print(included)
        if worst_pval > threshold_out:
            changed=True
            worst_feature = included[np.argmax(pvalues)]
            included.remove(worst_feature)
        if not changed:
            return included
    

backward_regression(X, y)
# -



