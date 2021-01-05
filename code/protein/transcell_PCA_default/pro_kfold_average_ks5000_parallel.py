#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:57:32 2020

@author: shanjuyeh
"""

import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import LeakyReLU
import matplotlib
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
import seaborn as sns
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.layers.advanced_activations import LeakyReLU
import random
import matplotlib.pyplot as plt
import h5py
from scipy.stats import iqr
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from sklearn.preprocessing import normalize
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
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
data_meta    = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/octad_cell_line_meta.csv')
data_matrix  = pd.read_hdf('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/data_matrix.h5', 'df')

types = list(set(data_feature['type']))
cell_line = data_matrix[['Unnamed: 0']]

# Take expression matrix
id_expression = data_feature.loc[data_feature['type'] == 'expression'].id
id_expression = id_expression.tolist()
expression_matrix = data_matrix.loc[:,id_expression]

# Take protein matrix
id_protein = data_feature.loc[data_feature['type'] == 'protein'].id
id_protein = id_protein.tolist()
protein_matrix = data_matrix.loc[:,id_protein]

# drop na
exp = expression_matrix.dropna(axis=0)
pro = protein_matrix.dropna(axis=0)
exp = exp.reset_index(drop=False)
pro = pro.reset_index(drop=False)
exp = exp.rename(columns = {'index' : 'Cell'})
pro = pro.rename(columns = {'index' : 'Cell'})

nona_matrix = pd.merge(exp, pro, on = 'Cell')

exp_x = nona_matrix.loc[:, id_expression]
pro_y = nona_matrix.loc[:, id_protein]
pro_names = pro_y.columns.to_list()

## take top ks 520
ks = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/data/ks2sample_TCGA.padj.csv')
#ks = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/ks2sample_TCGA.padj.csv')
lis_use = ks['genes'][0:5000]
exp_target = exp_x.loc[:, lis_use]

### load gscore name
#sampling = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/output/protein/other_models')
#sampling = sampling['Unnamed: 0']
##pro_names = sampling['Unnamed: 0']

## load pre-trained model
json_file = open('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.json', 'r')
#json_file = open('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks520_2step.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.h5")
#loaded_model.load_weights("/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks520_2step.h5")


from numpy.random import seed
seed(1) ##10
from tensorflow import set_random_seed
set_random_seed(1)

def transfer_evaluate_model(x_train, y_train, x_test, y_test):
    scaler_x = MinMaxScaler(feature_range=(0,1))
    X_train = pd.DataFrame(x_train)
    X_test = pd.DataFrame(x_test)
    scaler_x.fit(X_train)
    x_train = scaler_x.transform(X_train)
    x_test = scaler_x.transform(X_test)
    
    scaler_y = MinMaxScaler(feature_range=(-2, 2))
    Y_train = pd.DataFrame(y_train)
    Y_test = pd.DataFrame(y_test)
    scaler_y.fit(Y_train)
    y_train = scaler_y.transform(Y_train)
    y_test = scaler_y.transform(Y_test)

    a = loaded_model.output ### 10/22/2020
    x1 = Dense(512, activation = 'tanh', name = 'h1', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5))(a)
    x2 = Dropout(0.55, name = 'd1')(x1)
    x3 = Dense(224, activation = 'tanh', name = 'h2', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5))(x2)
    x4 = Dropout(0.25, name = 'd2')(x3)
    x5 = Dense(96, activation = 'tanh', name = 'h3', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5))(x4)
    x6 = Dropout(0.4, name = 'd3')(x5)
    x7 = Dense(1, name = 'h4', activation='tanh')(x6)
    
    for layer in loaded_model.layers:
        layer.trainable = True
        
    new_model = Model(loaded_model.input, x7)
    opt = optimizers.Adam(lr=0.0001)
    new_model.compile(loss='mean_squared_error', optimizer= opt)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    history = History()
    new_model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[history, es], shuffle=True)
       
    pre = new_model.predict(x_test)
    pre1 = scaler_y.inverse_transform(pre)    
    mse = mean_squared_error(Y_test,pre1)
    pearsonr = scipy.stats.pearsonr(Y_test, pre1)
    spearmanr = scipy.stats.spearmanr(Y_test, pre1.flatten())
    rmse = sqrt(mean_squared_error(Y_test, pre1.flatten()))
    r_square = r2_score (Y_test, pre1)
    Pearsonr = float(pearsonr[0])
    Pearsonr_pvalue = float(pearsonr[1])
    Spearmanr = float(spearmanr[0])
    Spearmanr_pvalue = float(spearmanr[1])
    total_1 = [rmse, mse, Pearsonr, Pearsonr_pvalue,  Spearmanr, Spearmanr_pvalue,  r_square]
    
    return total_1


def default_init_evaluate_model(x_train, y_train, x_test, y_test):
    scaler_x = MinMaxScaler(feature_range=(0,1))
    X_train = pd.DataFrame(x_train)
    X_test = pd.DataFrame(x_test)
    scaler_x.fit(X_train)
    x_train = scaler_x.transform(X_train)
    x_test = scaler_x.transform(X_test)
    
    scaler_y = MinMaxScaler(feature_range=(-2, 2))
    Y_train = pd.DataFrame(y_train)
    Y_test = pd.DataFrame(y_test)
    scaler_y.fit(Y_train)
    y_train = scaler_y.transform(Y_train)
    y_test = scaler_y.transform(Y_test)
    
    ### random intialization    
    new_model = Sequential()
    new_model.add(Dense(512, input_dim = exp_target.shape[1]))
    new_model.add(LeakyReLU(alpha=0.1))
    new_model.add(Dense(200))
    new_model.add(LeakyReLU(alpha=0.1))
    new_model.add(Dense(units=512, input_dim=x_train.shape[1], activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.55))
    new_model.add(Dense(units=224,activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.25))
    new_model.add(Dense(units=96,activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.4))
    new_model.add(Dense(units=1, activation='tanh'))    
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    opt = optimizers.Adam(lr=0.0001)
    new_model.compile(loss='mean_squared_error', optimizer=opt)
    new_model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[es])    
       
    pre = new_model.predict(x_test)
    pre1 = scaler_y.inverse_transform(pre)    
    mse = mean_squared_error(Y_test,pre1)
    pearsonr = scipy.stats.pearsonr(Y_test, pre1)
    spearmanr = scipy.stats.spearmanr(Y_test, pre1.flatten())
    rmse = sqrt(mean_squared_error(Y_test, pre1.flatten()))
    r_square = r2_score (Y_test, pre1)
    Pearsonr = float(pearsonr[0])
    Pearsonr_pvalue = float(pearsonr[1])
    Spearmanr = float(spearmanr[0])
    Spearmanr_pvalue = float(spearmanr[1])
    total_1 = [rmse, mse, Pearsonr, Pearsonr_pvalue,  Spearmanr, Spearmanr_pvalue,  r_square]
    
    return total_1

def pca_evaluate_model(x_train, y_train, x_test, y_test):    
    
    PCA_X = PCA(n_components=200)
    PCA_X.fit(x_train)
    X_train = PCA_X.transform(x_train)
    X_test = PCA_X.transform(x_test)
    
    scaler_y = MinMaxScaler(feature_range=(-2, 2))
    Y_train = pd.DataFrame(y_train)
    Y_test = pd.DataFrame(y_test)
    scaler_y.fit(Y_train)
    y_train = scaler_y.transform(Y_train)
    y_test = scaler_y.transform(Y_test)  
    
    ### the same structure with random initialization
    new_model = Sequential()
    new_model.add(Dense(units=512, input_dim=X_train.shape[1], activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.55))
    new_model.add(Dense(units=224,activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.25))
    new_model.add(Dense(units=96,activation='tanh', activity_regularizer=regularizers.l2(1e-3), bias_regularizer=regularizers.l2(1e-5)))
    new_model.add(Dropout(0.4))
    new_model.add(Dense(units=1, activation='tanh'))    
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    opt = optimizers.Adam(lr=0.0001)
    new_model.compile(loss='mean_squared_error', optimizer=opt)
    new_model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[es])     

    pre = new_model.predict(X_test)
    pre1 = scaler_y.inverse_transform(pre)    
    mse = mean_squared_error(Y_test,pre1)
    pearsonr = scipy.stats.pearsonr(Y_test, pre1)
    spearmanr = scipy.stats.spearmanr(Y_test, pre1.flatten())
    rmse = sqrt(mean_squared_error(Y_test, pre1.flatten()))
    r_square = r2_score (Y_test, pre1)
    Pearsonr = float(pearsonr[0])
    Pearsonr_pvalue = float(pearsonr[1])
    Spearmanr = float(spearmanr[0])
    Spearmanr_pvalue = float(spearmanr[1])
    total_1 = [rmse, mse, Pearsonr, Pearsonr_pvalue,  Spearmanr, Spearmanr_pvalue,  r_square]
    
    return total_1

methods = ['transfer', 'default_init', 'PCA']
#name = 'protein_14-3-3_epsilon_Caution'



kf = KFold(n_splits=5)
for method in methods:
   
    if method == 'transfer': 

       X = exp_target.values
       Y = pro_y[name].values
       out = []
       for train_index, test_index in kf.split(X):
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = Y[train_index], Y[test_index]
           total = transfer_evaluate_model(X_train, y_train, X_test, y_test)
           out.append(total)
       
       res = np.array(out)
       p_avg = list(np.mean(res, axis=0))
       p_std = list(np.std(res, axis=0))
       print("%s,avg_transfer_pro,%f,%f,%f,%e,%f,%e,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3], p_avg[4], p_avg[5], p_avg[6]))
       print("%s,std_transfer_pro,%f,%f,%f,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3], p_std[4], p_std[5], p_std[6]))

    if method == 'default_init':
        
        X = exp_target.values
        Y = pro_y[name].values
        out = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = default_init_evaluate_model(X_train, y_train, X_test, y_test)
            out.append(total)

        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_default_pro,%f,%f,%f,%e,%f,%e,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3], p_avg[4], p_avg[5], p_avg[6]))
        print("%s,std_default_pro,%f,%f,%f,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3], p_std[4], p_std[5], p_std[6]))

    
    if method == 'PCA':
        
        X = exp_x.values
        Y = pro_y[name].values
        out = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            total = pca_evaluate_model(X_train, y_train, X_test, y_test)
            out.append(total)

        res = np.array(out)
        p_avg = list(np.mean(res, axis=0))
        p_std = list(np.std(res, axis=0))
        print("%s,avg_pca_pro,%f,%f,%f,%e,%f,%e,%f\n" % (name, p_avg[0], p_avg[1], p_avg[2],p_avg[3], p_avg[4], p_avg[5], p_avg[6]))
        print("%s,std_pca_pro,%f,%f,%f,%f,%f,%f,%f\n" % (name, p_std[0], p_std[1], p_std[2],p_std[3], p_std[4], p_std[5], p_std[6]))

        
    

