#!/usr/bin/env python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.cross_validation import train_test_split


cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# utility function to plot the decision surface
def plot_surface(est, x_1, x_2, ax=None, threshold=0.0, contourf=False):
    """Plots the decision surface of ``est`` on features ``x1`` and ``x2``. """
    xx1, xx2 = np.meshgrid(np.linspace(x_1.min(), x_1.max(), 100), 
                           np.linspace(x_2.min(), x_2.max(), 100))

    # plot the hyperplane by evaluating the parameters on the grid
    X_pred = np.c_[xx1.ravel(), xx2.ravel()]                                        # convert 2d grid into seq of points
    if hasattr(est, 'predict_proba'):                                               # check if ``est`` supports probabilities
        pred = est.predict_proba(X_pred)[:, 1]                                      # take probability of positive class
    else:
        pred = est.predict(X_pred)
    Z = pred.reshape((100, 100))                                                    # reshape seq to grid
    if ax is None:
        ax = plt.gca()
    if contourf:
        ax.contourf(xx1, xx2, Z, levels=np.linspace(0, 1.0, 10), cmap=plt.cm.RdBu, alpha=0.6)
    ax.contour(xx1, xx2, Z, levels=[threshold], colors='black')
    ax.set_xlim((x_1.min(), x_1.max()))
    ax.set_ylim((x_2.min(), x_2.max()))
    plt.show()
    
def plot_datasets(est=None):
    """Plots the decision surface of ``est`` on each of the three datasets. """
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for (name, ds), ax in zip(datasets.items(), axes):
        X_train = ds['X_train']
        y_train = ds['y_train']
        X_test = ds['X_test']
        y_test = ds['y_test']
        
        # plot test lighter than training
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)         # Plot the training points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6) # and testing points
        ax.set_xlim(X_train[:, 0].min(), X_train[:, 0].max())                       # plot limits
        ax.set_ylim(X_train[:, 1].min(), X_train[:, 1].max())
        ax.set_xticks(())                                                           # no ticks
        ax.set_yticks(())                                                           # no ticks
        ax.set_ylabel('$x_1$')
        ax.set_xlabel('$x_0$')
        ax.set_title(name)
        if est is not None:
            est.fit(X_train, y_train)
            plot_surface(est, X_train[:, 0], X_train[:, 1], ax=ax, threshold=0.5, contourf=True)
            err = (y_test != est.predict(X_test)).mean()
            ax.text(0.88, 0.02, '%.2f' % err, transform=ax.transAxes)

def confusion_matrix(y_test, y_pred):
    cm = sk_confusion_matrix(y, y_pred)
    cm = pd.DataFrame(data=cm, columns=[-1, 1], index=[-1, 1])
    cm.columns.name = 'Predicted label'
    cm.index.name = 'True label'
    error_rate = (y_pred != y).mean()
    print('error rate: %.2f' % error_rate)
    return cm

# read data
df = pd.read_csv('https://d1pqsl2386xqi9.cloudfront.net/notebooks/Default.csv', index_col=0)

# downsample negative cases -- there are many more negatives than positives
indices = np.where(df.default == 'No')[0]
np.random.RandomState(13).shuffle(indices)
n_pos = (df.default == 'Yes').sum()
df = df.drop(df.index[indices[n_pos:]])

# get feature/predictor matrix as numpy array
X = df[['balance', 'income']].values

# encode class labels
classes, y = np.unique(df.default.values, return_inverse=True)
y = (y * 2) - 1  # map {0, 1} to {-1, 1}

# # fit OLS regression 
# est = LinearRegression(fit_intercept=True, normalize=True)
# est.fit(X, y)

# # plot data and decision surface
# ax = plt.gca()
# ax.scatter(df.balance, df.income, c=(df.default == 'Yes'), cmap=cm_bright)
# plot_surface(est, X[:, 0], X[:, 1], ax=ax)

# # the larger operator will return a boolean array which we will cast as integers for fancy indexing
# y_pred = (2 * (est.predict(X) > 0.0)) - 1
# print confusion_matrix(y, y_pred)

# create 80%-20% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
est = LinearRegression().fit(X_train, y_train) # fit on training data
y_pred = (2 * (est.predict(X) > 0.0)) - 1 # test on data that was not used for fitting
print confusion_matrix(y_test, y_pred)
