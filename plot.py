import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif, mutual_info_classif, mutual_info_regression

from data_processing import SEED

##############################################
##############################################
##############################################

def regplot(X, y, method="pearson", remove_na=False, **kwargs):
    X = X.copy()
    y = y.copy()
    
    if remove_na:
        X, y = X[~( X.isna()|y.isna() )], y[~( X.isna()|y.isna() )]
    
    cc = pd.concat([X, y], axis=1).corr(method=method).values[0][1]
    mi = mutual_info_regression(X.values.reshape(-1, 1), y, random_state=SEED)[0]
    
    sns.regplot(x=X, y=y, scatter_kws=dict(edgecolors="w"), **kwargs)
    plt.title(f"Correlation Coefficient: {cc:.5f}\nMutual Information: {mi:.5f}", fontsize="large")
    plt.tight_layout()

##############################################
##############################################
##############################################    
    
def multi_regplot(X, y, cols, method="pearson"):
    nrow = (len(cols)//3) + 1
    ncol = len(cols) if len(cols)<3 else 3
    fig = plt.figure(figsize=(18, 4.5*nrow))
    for i, col in enumerate(cols):
        ax = fig.add_subplot(nrow, np.max([2, ncol]), i+1)
        regplot(X[col], y, method=method, color=sns.color_palette()[i], ax=ax)
    if ncol == 1:
        fig.add_subplot(nrow, np.max([2, ncol]), i+2)
        plt.axis("off")
    plt.tight_layout()

##############################################
##############################################
##############################################    
    
def catplot(X, y):
    f, p = f_classif(y.values.reshape(-1, 1), X.astype(str))
    mi = mutual_info_classif(y.values.reshape(-1, 1), X.astype(str), random_state=SEED)[0]
    
    num_cat = X.nunique()
    rotation = 90 if num_cat>=7 else 0
    fig = plt.figure(figsize=(18, 4.5))
    
    plt.subplot(121)
    sns.boxplot(x=X, y=y, flierprops={"alpha": 0.5})
    plt.xticks(rotation=rotation)
    
    plt.subplot(122)
    ax = sns.countplot(x=X)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=rotation)
    
    fig.suptitle(f"F-Statistic: {f[0]:.5f}, P-Value: {p[0]:.5f}\nMutual Information: {mi:.5f}")
    plt.tight_layout()

##############################################
##############################################
##############################################   
    
def multi_catplot(X, y, cols):
    for col in cols:
        catplot(X[col], y)
    plt.tight_layout()