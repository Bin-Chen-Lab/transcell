#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:42:45 2020

@author: shanjuyeh
"""

import pandas as pd
import numpy as np
import csv
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import LeakyReLU
import matplotlib
import random
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy
import statistics
import keras
from keras import optimizers
from keras.optimizers import SGD
from sklearn.metrics import r2_score
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import concatenate
from keras import Sequential
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns, numpy as np
from keras.models import load_model
from keras import backend as K
from keras.callbacks import History
from math import sqrt
from keras import backend
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from itertools import product
import statistics
from scipy.stats import norm
from scipy.stats import skewnorm
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, \
    confusion_matrix, matthews_corrcoef, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from numpy import arange
import os
import sys


name = ' '.join(sys.argv[1:])
print("Processing %s" % name)

## mac
#data_feature = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/octad_cell_line_features.csv')
#data_meta = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/octad_cell_line_meta.csv')
#data_matrix = pd.read_hdf('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/data_matrix.h5', 'df')

# server
data_feature = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/octad_cell_line_features.csv')
data_meta = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/octad_cell_line_meta.csv')
data_matrix = pd.read_hdf('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/data_matrix.h5', 'df')

types = list(set(data_feature['type']))
cell_line = data_matrix[['Unnamed: 0']]

# Take expression matrix
id_expression = data_feature.loc[data_feature['type'] == 'expression'].id
id_expression = id_expression.tolist()
expression_matrix = data_matrix.loc[:,id_expression]

# Take mutation matrix
id_mutation = data_feature.loc[data_feature['type'] == 'mutation'].id
id_mutation = id_mutation.tolist()
mutation_matrix = data_matrix.loc[:,id_mutation]

# drop na
exp = expression_matrix.dropna(axis=0)
mu = mutation_matrix.dropna(axis=0)
exp = exp.reset_index(drop=False)
mu = mu.reset_index(drop=False)
exp = exp.rename(columns = {'index' : 'Cell'})
mu = mu.rename(columns = {'index' : 'Cell'})
nona_matrix = pd.merge(exp, mu, on = 'Cell')
exp_x = nona_matrix.loc[:, id_expression]
mu_y = nona_matrix.loc[:, id_mutation]
mu_names = mu_y.columns.to_list()

## Find mutation proportion over 10%
mu_tmp = mu_y.values
mu_tmp1 = mu_tmp.sum(axis=0)
mu_count = pd.DataFrame(mu_tmp1, index = mu_names)
mu_count = mu_count.sort_values(by=[0], ascending = False)
mu_count = mu_count.rename(columns={0:'count'})
mu_name = mu_count.index.tolist()
mu_top_name = mu_name[:20]


def LOGISTIC_LASSO_FEATURE(exp_x, mu_name):
    y = mu_y[mu_name]
    scaler=StandardScaler()
    scaler.fit(exp_x)
    exp_xx = scaler.transform(exp_x)
    sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=500))
    sel.fit(exp_xx, y)
    lasso_x = sel.transform(exp_xx)
    y = y.to_numpy()
    
    return lasso_x, y


def LOGISTIC_LASSO(x_train, y_train, x_test, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred , average="weighted")
    recall = recall_score(y_test, y_pred , average="weighted")
    
    total = [f1, auc, precision, recall]
    
    return total


def LOGISTIC_ELASTIC_FEATURE(exp_x, mu_name):
    y = mu_y[mu_name]
    scaler = StandardScaler()
    scaler.fit(exp_x)
    exp_xx = scaler.transform(exp_x)
    sel = SelectFromModel(LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=5000))
    sel.fit(exp_xx, y)
    elastic_x = sel.transform(exp_xx)
    y = y.to_numpy()
    
    return elastic_x, y
    


def LOGISTIC_ELASTIC(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter = 5000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred , average="weighted")
    recall = recall_score(y_test, y_pred , average="weighted")
    
    total = [f1, auc, precision, recall]
    
    return total


def RF_FEATURE(exp_x, mu_name):
    y = mu_y[mu_name]
    sel = SelectFromModel(RandomForestClassifier(n_estimators=500, max_depth=300), threshold=10**-3)
    #sel = SelectFromModel(RandomForestClassifier(n_estimators=500, max_depth=300))
    sel.fit(exp_x, y)
    rf_x = sel.transform(exp_x)
    y = y.to_numpy()
    
    return rf_x, y


def RF(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=80)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred , average="weighted")
    recall = recall_score(y_test, y_pred , average="weighted")

    total = [f1, auc, precision, recall]
    
    return total


#methods = ['LOGISTIC_LASSO', 'LOGISTIC_ELASTIC', 'RF']
methods = ['LOGISTIC_LASSO']


kf = KFold(n_splits=5)
for method in methods:
   
    if method == 'LOGISTIC_LASSO':
        out =[]
        lasso_X, Y = LOGISTIC_LASSO_FEATURE(exp_x, name)
        for train_index, test_index in kf.split(lasso_X):
            X_train, X_test = lasso_X[train_index], lasso_X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = LOGISTIC_LASSO(X_train, y_train, X_test, y_test)
            out.append(total)
        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_logisticLasso_mu,%f,%f,%f,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3]))
        print("%s,std_logisticLasso_mu,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3]))
        
        
    elif method == 'LOGISTIC_ELASTIC':
        out =[]
        elastic_X, Y = LOGISTIC_ELASTIC_FEATURE(exp_x, name)
        for train_index, test_index in kf.split(elastic_X):
            X_train, X_test = elastic_X[train_index], elastic_X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = LOGISTIC_ELASTIC(X_train, y_train, X_test, y_test)
            out.append(total)
        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_logisticElastic_mu,%f,%f,%f,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3]))
        print("%s,std_logisticElastic_mu,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3]))
        
    else:
        out = []
        rf_X, Y = RF_FEATURE(exp_x, name)
        for train_index, test_index in kf.split(rf_X):
            X_train, X_test = rf_X[train_index], rf_X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = RF(X_train, y_train, X_test, y_test)
            out.append(total)
        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_rf_mu,%f,%f,%f,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3]))
        print("%s,std_rf_mu,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3]))
        
       
    
    




