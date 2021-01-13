#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:57:27 2021

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

ks = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/mutation/ks2sample_TCGA.padj.csv')
#ks = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/data/ks2sample_TCGA.padj.csv')
lis_use = ks['genes'][0:5000]
exp_target = exp_x.loc[:, lis_use] ## we take the last row to be the new cell line as an example
lis_use_df = pd.DataFrame(lis_use)


##### Mapping new data to 5000 KS features
data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/mutation/' + dataset + '.csv')
data = data[["Unnamed: 0", new_cell_line_name]]
gene = data['Unnamed: 0']

#data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/mutation/exprs_newCellLine_mutation.csv')
#data = data[["Unnamed: 0", "pqr"]]
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

X = exp_target[:1202]
Y = mu_y[feature_name][:1202]

json_file = open('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/mutation/encoder_ks5000_2step.json', 'r')
#json_file = open('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.h5')
loaded_model.load_weights('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/Further_application/mutation/encoder_ks5000_2step.h5')

from numpy.random import seed
seed(20)
from tensorflow import set_random_seed
set_random_seed(20)


def TransCell_mutation_model(X, Y, new_input):
    scaler_x = MinMaxScaler(feature_range=(0,1)) ## new1 / 0930
    X_df = pd.DataFrame(X)
    new_input_df = pd.DataFrame(new_input)
    scaler_x.fit(X_df)
    XX = scaler_x.transform(X_df)
    New_input = scaler_x.transform(new_input_df)
                
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
    new_model.fit(XX, Y, validation_split=0.1, epochs=100, batch_size=64, verbose=1, callbacks=[history, es], shuffle=True)
    
    pre = new_model.predict(New_input)    
    pre1 = (pre>0.5)

    return pre1

pre1 = TransCell_mutation_model(X, Y, new_input)
print('New cell line (%s) prediction result for %s: %s (False: non-mutation / True: mutation)' % (new_cell_line_name, feature_name, pre1.tolist()[0][0]))


