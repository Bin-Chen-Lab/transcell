#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:35:21 2021

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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_name', type=str)
parser.add_argument('--new_cell_line_name', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

feature_name = args.feature_name
new_cell_line_name = args.new_cell_line_name
dataset = args.dataset

# mac
data_feature = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/octad_cell_line_features.csv')
data_meta = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/octad_cell_line_meta.csv')
data_matrix = pd.read_hdf('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/data_matrix.h5', 'df')

types = list(set(data_feature['type']))
cell_line = data_matrix[['Unnamed: 0']]

# Take expression matrix
id_expression = data_feature.loc[data_feature['type'] == 'expression'].id
id_expression = id_expression.tolist()
expression_matrix = data_matrix.loc[:,id_expression]

# Take cn matrix
id_cn = data_feature.loc[data_feature['type'] == 'cn'].id
id_cn = id_cn.tolist()
cn_matrix = data_matrix.loc[:,id_cn]

# drop na
exp = expression_matrix.dropna(axis=0)
cn = cn_matrix.dropna(axis=0)
exp = exp.reset_index(drop=False)
cn = cn.reset_index(drop=False)
exp = exp.rename(columns = {'index' : 'Cell'})
cn = cn.rename(columns = {'index' : 'Cell'})

nona_matrix = pd.merge(exp, cn, on = 'Cell')

exp_x = nona_matrix.loc[:, id_expression]
cn_y = nona_matrix.loc[:, id_cn]
cn_names = cn_y.columns.to_list()

ks = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/cnv/ks2sample_TCGA.padj.csv')
#ks = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/data/ks2sample_TCGA.padj.csv')
lis_use = ks['genes'][0:5000]
exp_target = exp_x.loc[:, lis_use] ## we take the last row to be the new cell line as an example
lis_use_df = pd.DataFrame(lis_use)


##### Mapping new data to 5000 KS features
data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/cnv/' + dataset + '.csv')
data = data[["Unnamed: 0", new_cell_line_name]]
gene = data['Unnamed: 0']

#data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/cnv/exprs_newCellLine_cnv.csv')
#data = data[["Unnamed: 0", "mno"]]
#gene = data['Unnamed: 0']

lis=[]
for i in range(len(gene)):
   tmp = 'expression_' + gene[i]
   lis.append(tmp)

lis_s = pd.Series(lis)
data.insert(0, 'genes', lis_s)
data = data.drop(['Unnamed: 0'], axis=1)
data_map = pd.merge(data, lis_use_df, on='genes')
new_input = pd.DataFrame(data_map[new_cell_line_name]).T

X = exp_target[:1195]
Y = cn_y[feature_name][:1195]

json_file = open('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/cnv/encoder_ks5000_2step.json', 'r')
#json_file = open('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.h5')
loaded_model.load_weights('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/cnv/encoder_ks5000_2step.h5')

from numpy.random import seed
seed(18)
from tensorflow import set_random_seed
set_random_seed(18)

def TransCell_cnv_model(X, Y, new_input):

    scaler_x = MinMaxScaler(feature_range=(0,1)) 
    X_df = pd.DataFrame(X)
    new_input_df = pd.DataFrame(new_input)
    scaler_x.fit(X_df)
    XX = scaler_x.transform(X_df)
    New_input = scaler_x.transform(new_input_df)
            
    scaler_y = MinMaxScaler(feature_range=(-1,1))
    Y_df = pd.DataFrame(Y)
    scaler_y.fit(Y_df)
    YY = scaler_y.transform(Y_df)
   
    
    ### 10/22/2020 
    a = loaded_model.output
    x1 = Dense(256, activation = 'tanh', name = 'h1', activity_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-5))(a)
    x2 = Dropout(0.6)(x1)
    x3 = Dense(256, activation = 'tanh', name = 'h2', activity_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-5))(x2)
    x4 = Dropout(0.35, name = 'd1')(x3)
    x5 = Dense(32, activation = 'tanh', name = 'h3', activity_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-5))(x4)
    x6 = Dropout(0.55, name = 'd2')(x5)
    x7 = Dense(1, name = 'h4', activation = 'tanh')(x6)
    
    for layer in loaded_model.layers:
        layer.trainable = True
        
    new_model = Model(loaded_model.input, x7)
    opt = optimizers.Adam(lr=0.001)
    new_model.compile(loss='mean_squared_error', optimizer= opt)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    history = History()
    new_model.fit(XX, YY, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[history, es], shuffle=True)

    pre = new_model.predict(New_input)
    pre1 = scaler_y.inverse_transform(pre)    
    
    return pre1

pre1 = TransCell_cnv_model(X, Y, new_input)
print('New cell line (%s) prediction result for %s: %.3f' % (new_cell_line_name, feature_name, pre1.tolist()[0][0]))







