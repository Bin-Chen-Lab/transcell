#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:58:30 2020

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
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from scipy.stats import iqr
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from sklearn.preprocessing import normalize
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.regularizers import l2
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

name = ' '.join(sys.argv[1:])
print("Processing %s" % name)


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

## take top ks 5000
ks = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/data/ks2sample_TCGA.padj.csv')
lis_use = ks['genes'][0:5000]
exp_target = exp_x.loc[:, lis_use]


## load pre-trained model
json_file = open('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.h5')

from numpy.random import seed
seed(20)
from tensorflow import set_random_seed
set_random_seed(20)


def transfer_evaluate_model(x_train, y_train, x_test, y_test):
    scaler_x = MinMaxScaler(feature_range=(0,1)) ## new1 / 0930
    X_train = pd.DataFrame(x_train)
    X_test = pd.DataFrame(x_test)
    scaler_x.fit(X_train)
    x_train = scaler_x.transform(X_train)
    x_test = scaler_x.transform(X_test)
                
    a = loaded_model.output
    x1 = Dense(416, activation = 'relu', name = 'h1', kernel_regularizer=l2(0.01))(a)
    x2 = Dropout(0.05, name='d1')(x1)
    x3 = Dense(192, activation = 'relu', name = 'h2', kernel_regularizer=l2(0.01))(x2)
    x4 = Dropout(0.3, name = 'd2')(x3)
    x5 = Dense(32, activation = 'relu', name = 'h3', kernel_regularizer=l2(0.01))(x4)
    x6 = Dropout(0.05, name = 'd3')(x5)
    x7 = Dense(1, name = 'h4', activation = 'sigmoid')(x6)
    
    for layer in loaded_model.layers:
        layer.trainable = True
        
    new_model = Model(loaded_model.input, x7)
    opt = optimizers.Adam(lr=0.001)
    new_model.compile(loss='binary_crossentropy', optimizer= opt)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    history = History()
    new_model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[history, es], shuffle=True)
    
    pre = new_model.predict(x_test)    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pre)
    auc = metrics.auc(fpr, tpr)
    
    #y_pred = np.argmax(pre, axis=1)
    y_pred = (np.asarray(pre)).round()
    
    precision = precision_score(y_test, y_pred , average="weighted")
    recall = recall_score(y_test, y_pred , average="weighted")
    f1 = f1_score(y_test, y_pred , average="weighted")
    
    total = [auc, precision, recall, f1]

    return total


def default_init_evaluate_model(x_train, y_train, x_test, y_test):
    scaler_x = MinMaxScaler(feature_range=(0,1))
    X_train = pd.DataFrame(x_train)
    X_test = pd.DataFrame(x_test)
    scaler_x.fit(X_train)
    x_train = scaler_x.transform(X_train)
    x_test = scaler_x.transform(X_test)
    
    ### random intialization    
    new_model = Sequential()
    new_model.add(Dense(512, input_dim = exp_target.shape[1]))
    new_model.add(LeakyReLU(alpha=0.1))
    new_model.add(Dense(200))
    new_model.add(LeakyReLU(alpha=0.1))
    new_model.add(Dense(units=416, input_dim=x_train.shape[1], activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.05))
    new_model.add(Dense(units=192,activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.3))
    new_model.add(Dense(units=32,activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.05))
    new_model.add(Dense(units=1, activation='tanh'))    
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    opt = optimizers.Adam(lr=0.001)
    new_model.compile(loss='mean_squared_error', optimizer=opt)
    new_model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[es])    
       
    pre = new_model.predict(x_test)    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pre)
    auc = metrics.auc(fpr, tpr)
    
    #y_pred = np.argmax(pre, axis=1)
    y_pred = (np.asarray(pre)).round()
    precision = precision_score(y_test, y_pred , average="weighted")
    recall = recall_score(y_test, y_pred , average="weighted")
    f1 = f1_score(y_test, y_pred , average="weighted")
    total = [auc, precision, recall, f1]
    
    return total

def pca_evaluate_model(x_train, y_train, x_test, y_test):    
    
    PCA_X = PCA(n_components=200)
    PCA_X.fit(x_train)
    X_train = PCA_X.transform(x_train)
    X_test = PCA_X.transform(x_test)
    
    ### the same structure with random initialization
    new_model = Sequential()
    new_model.add(Dense(units=416, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    new_model.add(Dropout(0.05))
    new_model.add(Dense(units=192,activation='relu', kernel_regularizer=l2(0.01)))
    new_model.add(Dropout(0.3))
    new_model.add(Dense(units=32,activation='relu', kernel_regularizer=l2(0.01)))
    new_model.add(Dropout(0.05))
    new_model.add(Dense(units=1, activation='sigmoid'))    
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 50)
    opt = optimizers.Adam(lr=0.001)
    new_model.compile(loss='mean_squared_error', optimizer=opt)
    new_model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[es])     

    pre = new_model.predict(X_test)    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pre)
    auc = metrics.auc(fpr, tpr)
    
    #y_pred = np.argmax(pre, axis=1)
    y_pred = (np.asarray(pre)).round
    
    precision = precision_score(y_test, y_pred , average="weighted")
    recall = recall_score(y_test, y_pred , average="weighted")
    f1 = f1_score(y_test, y_pred , average="weighted")
    total = [auc, precision, recall, f1]
    
    return total

methods = ['transfer', 'default_init', 'PCA']

kf = KFold(n_splits=5)
for method in methods:
   
    if method == 'transfer':
      
       X = exp_target.values
       Y = mu_y[name].values
       out=[]
       for train_index, test_index in kf.split(X):
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = Y[train_index], Y[test_index]
           total = transfer_evaluate_model(X_train, y_train, X_test, y_test)
           out.append(total)
       
       res = np.array(out)
       p_avg = list(np.mean(res, axis=0))
       p_std = list(np.std(res, axis=0))
       print("%s,avg_transfer_mu,%f,%f,%f,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3]))
       print("%s,std_transfer_mu,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3]))

       
    if method == 'default_init':
        
        X = exp_target.values
        Y = mu_y[name].values
        out = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = default_init_evaluate_model(X_train, y_train, X_test, y_test)
            out.append(total)

        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_default_mu,%f,%f,%f,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3]))
        print("%s,std_default_mu,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3]))

        
    if method == 'PCA':
        
        X = exp_x.values
        Y = mu_y[name].values
        out = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = pca_evaluate_model(X_train, y_train, X_test, y_test)
            out.append(total)

        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_pca_mu,%f,%f,%f,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3]))
        print("%s,std_pca_mu,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3]))

       













