#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:57:37 2020

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
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import h5py
from scipy.stats import iqr
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json


## TCGA PAN cancer data (all of them are cancer patients' data)
f = h5py.File('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/OCTAD_pan_cancer_TCGA/pan.cancer.TCGA.h5', 'r')
list(f.keys())
data_matrix = f.get('tpm.matrix')
data_np = np.array(data_matrix)
data_df = pd.DataFrame(data_np)
col = pd.Series(f.get('gene.name'))
row = pd.Series(f.get('sample.name'))
## decode
COL=[]
for i in range(len(col)):
    tmp=col[i].decode("utf-8")
    COL.append(tmp)
COL = pd.Series(COL)

ROW=[]
for i in range(len(row)):
    tmp = row[i].decode("utf-8")
    ROW.append(tmp)
ROW = pd.Series(ROW)

data_df = data_df.T
data_df.columns=COL
data_df.index = ROW

####### OCTAD CELL LINES
## mac
data_feature = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/octad_cell_line_features.csv')
data_meta = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/octad_cell_line_meta.csv')
data_matrix = pd.read_hdf('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/data_matrix.h5', 'df')

## Take expression matrix
id_expression = data_feature.loc[data_feature['type'] == 'expression'].id
id_expression = id_expression.tolist()
expression_matrix = data_matrix.loc[:,id_expression]

# drop na
exp = expression_matrix.dropna(axis=0)
exp = exp.reset_index(drop=False)
exp = exp.rename(columns = {'index' : 'Cell'})
lis_new1=[]
for i in range(len(COL)):
    tmp = 'expression_' + COL[i]
    lis_new1.append(tmp)
exp_use = exp.loc[:, lis_new1]


#### Take KS genes
ks = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/ks2sample_TCGA.padj.csv')
lis = ks[0:5000]
lis_use = lis['genes']
lis_new=[]
for i in range(len(lis_use)):
    tmp = lis_use[i].split('expression_')[1]
    lis_new.append(tmp)
EXP_X = data_df.loc[:, lis_new]  ## TCGA
EXP_X_CellLine = exp_use.loc[:, lis_use] ## CellLine

#####======================== Final one
exp_x = StandardScaler().fit_transform(EXP_X)
ae_exp = Sequential()
ae_exp.add(Dense(512, input_dim = exp_x.shape[1]))
ae_exp.add(LeakyReLU(alpha=0.1))
ae_exp.add(Dense(200))
ae_exp.add(LeakyReLU(alpha=0.1))
ae_exp.add(Dense(512))
ae_exp.add(LeakyReLU(alpha=0.1))
ae_exp.add(Dense(5000))
ae_exp.compile(loss='mean_squared_error', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
history_exp = History()
ae_exp.fit(exp_x, exp_x, epochs = 100, shuffle = True, batch_size = 128, callbacks=[history_exp, es], validation_split=0.1)

#get_latent_output = K.function([ae_exp.layers[0].input], [ae_exp.layers[1].output]) # for encoder
#out = get_latent_output([exp_target])[0]
#out = out.tolist()
#encoded_exp = pd.DataFrame(data=out, index=pd.DataFrame(exp_target).index)
#encoded_exp = encoded_exp.to_numpy()

encoder = Model(ae_exp.input, ae_exp.layers[3].output)
# serialize model to JSON
model_json = encoder.to_json()
with open("/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks5000_pretrain.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder.save_weights("/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks5000_pretrain.h5")
print("Saved model to disk")

## load pre-trained model
json_file = open('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks5000_pretrain.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks5000_pretrain.h5")

exp_xcell = StandardScaler().fit_transform(EXP_X_CellLine)
a = loaded_model.output
x1 = Dense(512)(a)
x2 = LeakyReLU(alpha=0.1)(x1)
x3 = Dense(5000)(x2) 

new_model = Model(loaded_model.input, x3)
new_model.compile(loss='mean_squared_error', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
history_exp = History()
new_model.fit(exp_xcell, exp_xcell, epochs = 100, shuffle = True, batch_size = 128, callbacks=[history_exp, es], validation_split=0.1)

encoder_2stage = Model(new_model.input, new_model.layers[4].output)
# serialize model to JSON
model_json = encoder_2stage.to_json()
with open("/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks5000_2step.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder_2stage.save_weights("/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/encoder/encoder_ks5000_2step.h5")
print("Saved model to disk")










