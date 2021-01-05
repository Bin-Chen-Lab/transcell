#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:26:35 2020

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
from sklearn.linear_model import Lasso
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
from keras.layers.advanced_activations import LeakyReLU
import random
from scipy.stats import iqr
from sklearn.feature_selection import SelectFromModel
from keras.models import model_from_json

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

# Take gene_effect matrix
id_geneeffect = data_feature.loc[data_feature['type'] == 'gene_effect'].id
id_geneeffect = id_geneeffect.tolist()
geneeffect_matrix = data_matrix.loc[:,id_geneeffect]

# drop na
exp = expression_matrix.dropna(axis=0)
effect = geneeffect_matrix.dropna(axis=0)
exp = exp.reset_index(drop=False)
effect = effect.reset_index(drop=False)
exp = exp.rename(columns = {'index' : 'Cell'})
effect = effect.rename(columns = {'index' : 'Cell'})

nona_matrix = pd.merge(exp, effect, on = 'Cell')

exp_x = nona_matrix.loc[:, id_expression]
effect_y = nona_matrix.loc[:, id_geneeffect]
effect_names = effect_y.columns.to_list()


def LASSO_FEATURE(exp_x, effect_name):
    y = effect_y[effect_name]
    lasso = Lasso(alpha=0.03, max_iter=10000) 
    lasso.fit(exp_x, y)   
    Rank = pd.DataFrame(lasso.coef_ , index = exp_x.columns)
    Rank = Rank.reset_index(drop=False)
    Rank = Rank.rename(columns={'index':'feature', 0 : 'coef'})
    la_tmp = Rank[abs(Rank['coef'])>0]
    la_tmp = la_tmp.reset_index(drop=True)
    Rank_abs = [abs(i) for i in la_tmp.coef]
    Rank_abs = pd.DataFrame(Rank_abs)
    final = pd.concat([la_tmp[['feature']], Rank_abs], axis=1)
    final = final.rename(columns={0:'coef'})
    final = final.sort_values(by=['coef'],ascending= False)
    final = final.reset_index(drop=True)
    #lst = final.iloc[:num].feature.tolist()
    lst = final.feature.tolist()
    lasso_x = exp_x.loc[:,exp_x.columns.isin(lst)]
    lasso_x = lasso_x.to_numpy()
    
    return lasso_x, y


def LASSO(x_train, y_train, x_test, y_test):
    clf = Lasso(alpha=0.001,  max_iter=10000) 
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    pearsonr = scipy.stats.pearsonr(y_test, y_pred.flatten())
    spearmanr = scipy.stats.spearmanr(y_test, y_pred.flatten())
    rmse = sqrt(mean_squared_error(y_test, y_pred.flatten()))
    r_square = r2_score (y_test, y_pred)
    Pearsonr = float(pearsonr[0])
    Pearsonr_pvalue = float(pearsonr[1])
    Spearmanr = float(spearmanr[0])
    Spearmanr_pvalue = float(spearmanr[1])
    total = [rmse, mse, Pearsonr, Pearsonr_pvalue,  Spearmanr, Spearmanr_pvalue,  r_square]
    
    return total


def ELASTIC_FEATURE(exp_x, effect_name):
    y = effect_y[effect_name]
    elastic = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000) 
    elastic.fit(exp_x, y)
    coef_elastic = pd.Series(elastic.coef_, index = exp_x.columns)
    coef_elastic = coef_elastic.reset_index(drop=False)
    coef_elastic = coef_elastic.rename(columns={'index':'feature', 0:'coef'})
    e_tmp = coef_elastic[abs(coef_elastic['coef'])>0]
    Rank = e_tmp.reset_index(drop=True)
    Rank_abs = [abs(i) for i in Rank.coef]
    Rank_abs = pd.DataFrame(Rank_abs)
    final = pd.concat([Rank[['feature']], Rank_abs], axis=1)
    final = final.rename(columns={0:'coef'})
    final = final.sort_values(by=['coef'],ascending= False)
    final = final.reset_index(drop=True)
    #lst = final.iloc[:num].feature.tolist()
    lst = final.feature.tolist()
    elastic_x = exp_x.loc[:, exp_x.columns.isin(lst)]
    elastic_x = elastic_x.to_numpy()
    
    return elastic_x, y


def ELASTIC(x_train, y_train, x_test, y_test):
    clf = ElasticNet(alpha=0.01, l1_ratio = 0.3, max_iter=10000) 
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    pearsonr = scipy.stats.pearsonr(y_test, y_pred.flatten())
    spearmanr = scipy.stats.spearmanr(y_test, y_pred.flatten())
    rmse = sqrt(mean_squared_error(y_test, y_pred.flatten()))
    r_square = r2_score (y_test, y_pred)
    Pearsonr = float(pearsonr[0])
    Pearsonr_pvalue = float(pearsonr[1])
    Spearmanr = float(spearmanr[0])
    Spearmanr_pvalue = float(spearmanr[1])
    total = [rmse, mse, Pearsonr, Pearsonr_pvalue,  Spearmanr, Spearmanr_pvalue,  r_square]
    
    return total


def RF_FEATURE(exp_x, effect_name):
    y = effect_y[effect_name]
    rf = RandomForestRegressor(n_estimators= 500, max_depth=300) 
    rf.fit(exp_x, y)
    coef = pd.Series(rf.feature_importances_, index = exp_x.columns)
    chose_forest = pd.DataFrame(coef)
    chose_forest = chose_forest.reset_index(drop=False)
    chose_forest = chose_forest.rename(columns={'index' : 'feature', 0 : 'coef'})
    chose_forest = chose_forest.sort_values(by=['coef'],ascending= False)
    chose_forest = chose_forest.reset_index(drop=True)
    chose = chose_forest.loc[chose_forest['coef']>10**-4, ['feature']]
    lst = chose['feature'].tolist()
    rf_x = exp_x.loc[:, exp_x.columns.isin(lst)]
    rf_x = rf_x.to_numpy()
    
    return rf_x, y


def RF(x_train, y_train, x_test, y_test):
    rf = RandomForestRegressor(n_estimators=500, max_depth=80) 
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    pearsonr = scipy.stats.pearsonr(y_test, y_pred.flatten())
    spearmanr = scipy.stats.spearmanr(y_test, y_pred.flatten())
    rmse = sqrt(mean_squared_error(y_test, y_pred.flatten()))
    r_square = r2_score (y_test, y_pred)
    Pearsonr = float(pearsonr[0])
    Pearsonr_pvalue = float(pearsonr[1])
    Spearmanr = float(spearmanr[0])
    Spearmanr_pvalue = float(spearmanr[1])
    total = [rmse, mse, Pearsonr, Pearsonr_pvalue,  Spearmanr, Spearmanr_pvalue,  r_square]
    
    return total


methods = ['LASSO', 'ELASTIC', 'RF']


kf = KFold(n_splits=5)
for method in methods:
    
    if method == 'LASSO':
        out =[]
        lasso_X, Y = LASSO_FEATURE(exp_x, name)
        for train_index, test_index in kf.split(lasso_X):
            X_train, X_test = lasso_X[train_index], lasso_X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = LASSO(X_train, y_train, X_test, y_test)
            out.append(total)
        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_lasso_gscore,%f,%f,%f,%e,%f,%e,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3], p_avg[4], p_avg[5], p_avg[6]))
        print("%s,std_lasso_gscore,%f,%f,%f,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3], p_std[4], p_std[5], p_std[6]))
           
    elif method == 'ELASTIC':
        out =[]
        elastic_X, Y = ELASTIC_FEATURE(exp_x, name)
        for train_index, test_index in kf.split(elastic_X):
            X_train, X_test = elastic_X[train_index], elastic_X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = ELASTIC(X_train, y_train, X_test, y_test)
            out.append(total)
        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_elastic_gscore,%f,%f,%f,%e,%f,%e,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3], p_avg[4], p_avg[5], p_avg[6]))
        print("%s,std_elastic_gscore,%f,%f,%f,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3], p_std[4], p_std[5], p_std[6]))
        
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
        print("%s,avg_rf_gsocre,%f,%f,%f,%e,%f,%e,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3], p_avg[4], p_avg[5], p_avg[6]))
        print("%s,std_rf_gscore,%f,%f,%f,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3], p_std[4], p_std[5], p_std[6]))
























