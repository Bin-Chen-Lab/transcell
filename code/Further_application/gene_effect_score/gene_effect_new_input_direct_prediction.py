# -*- coding: utf-8 -*-

#This code uses previously saved 5-fold validation model weights in training to make direct prediction (re-training not required)

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
#from keras.layers.advanced_activations import LeakyReLU
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

os.system("taskset -p 0xff %d" % os.getpid())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

name = ' '.join(sys.argv[1:])
print("Processing %s" % name)


# load OCTAD server
data_feature = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/octad_cell_line_features.csv')
data_meta = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/octad_cell_line_meta.csv')
data_matrix = pd.read_hdf('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/protein_predict/data/data_matrix.h5', 'df')

# load external pediatric data server
data_ped =pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/RuoqiaoChen/DepMap_Public_20q1_Pediatric_Solid_Tumor_Subset/new_gene_expr.csv')
meta_ped = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/RuoqiaoChen/DepMap_Public_20q1_Pediatric_Solid_Tumor_Subset/new_cell_gscore_meta.csv')

types = list(set(data_feature['type']))
cell_line = data_matrix[['Unnamed: 0']]
# pediatric_cellLine = pd.Series(data_ped.index)
# pediatric_cellLine_df = pd.DataFrame(pediatric_cellLine)
# pediatric_cellLine_df.to_csv('F:\Project\GeneExp_prediction\data\Pediatric_Solid_Tumor\code\pediatric_cellLine.csv')

# Take expression matrix
id_expression = data_feature.loc[data_feature['type'] == 'expression'].id
id_expression = id_expression.tolist()
expression_matrix = data_matrix.loc[:,id_expression]

# Take gene effect matrix
id_gene_effect = data_feature.loc[data_feature['type'] == 'gene_effect'].id
id_gene_effect = id_gene_effect.tolist()
gene_effect_matrix = data_matrix.loc[:,id_gene_effect]

# drop na
exp = expression_matrix.dropna(axis=0)
exp = exp.reset_index(drop=False)
exp = exp.rename(columns = {'index' : 'Cell'})
#exp_target = exp.loc[:, top_lis_new] ## pull out 5000 top-varying genes
#exp.insert(0, 'Cell', exp['Cell'])
effect = gene_effect_matrix.dropna(axis=0)
effect = effect.reset_index(drop=False)
effect = effect.rename(columns = {'index' : 'Cell'})
nona_matrix = pd.merge(exp, effect, on = 'Cell')
exp_x = nona_matrix.loc[:, id_expression]
effect_y = nona_matrix.loc[:, id_gene_effect]
effect_names = gene_effect_matrix.columns.to_list()

### Preprocess ped data
data_ped = data_ped.set_index('Unnamed: 0')
data_ped_col = pd.Series(data_ped.columns)
new_col=[]
for i in range(len(data_ped_col)):
    tmp = data_ped_col[i].split(' ')[0]
    tmp_1 = 'expression_' + tmp
    new_col.append(tmp_1)

new_col = pd.Series(new_col)
data_ped.columns = new_col
ped_col = pd.DataFrame({'genes':new_col})

#import gene effect score true data
gene_effect_true = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/RuoqiaoChen/DepMap_Public_20q1_Pediatric_Solid_Tumor_Subset/new_true_gscore.csv')
gene_effect_true = gene_effect_true.set_index('Unnamed: 0')
ext_used = pd.merge(data_ped, gene_effect_true, right_index = True, left_index = True)
data_ped = ext_used[new_col]
data_ped = data_ped.reindex(gene_effect_true.index)

gene_effect_true_col = pd.Series(gene_effect_true.columns)
new_col=[]
for i in range(len(gene_effect_true_col)):
    tmp = gene_effect_true_col[i].split(' ')[0]
    tmp_1 = 'gene_effect_' + tmp
    new_col.append(tmp_1)

new_col = pd.Series(new_col)
gene_effect_true.columns = new_col
gene_effect_true_col = pd.DataFrame({'genes':new_col})


## take top ks 5000
#ks = pd.read_csv('F:\Project\GeneExp_prediction\data\external_MCF7\code\ks2sample_TCGA.padj.csv')
ks = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/data/ks2sample_TCGA.padj.csv')
lis_use = ks['genes'][0:5000]
exp_target = exp_x.loc[:, lis_use]
#data_map = pd.merge(lis_use_df, ped_col, on='genes') ### pediatric data genes all map 5000
new_input = data_ped.loc[:, lis_use]

#json_file = open('F:\Project\GeneExp_prediction\data\external_MCF7\code\encoder_ks5000_2step.json', 'r')
json_file = open('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/encoder/encoder_ks5000_2step.h5')

from numpy.random import seed
seed(6)
from tensorflow import set_random_seed
set_random_seed(6)
# from tensorflow import set_random_seed
# set_random_seed(10)
def transfer_evaluate_model(x_train, y_train, x_test, y_test, new_input, y_ext):

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    X_train = pd.DataFrame(x_train)
    X_test = pd.DataFrame(x_test)
    scaler_x.fit(X_train)
    x_train = scaler_x.transform(X_train)
    x_test = scaler_x.transform(X_test)
    x_ext = scaler_x.transform(new_input)
            
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    Y_train = pd.DataFrame(y_train)
    Y_test = pd.DataFrame(y_test)
    scaler_y.fit(Y_train)
    y_train = scaler_y.transform(Y_train)
    y_test = scaler_y.transform(Y_test)
    

##  pre-trained encoder     
    a = loaded_model.output
    x1 = Dense(288, activation = 'tanh', name = 'h1', activity_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-5))(a)
    x2 = Dropout(0.6, name = 'd1')(x1)
    x3 = Dense(128, activation = 'tanh', name = 'h2', activity_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-5))(x2)
    x4 = Dropout(0.35, name = 'd2')(x3)
    x5 = Dense(64, activation = 'tanh', name = 'h3', activity_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-5))(x4)
    x6 = Dropout(0.2, name = 'd3')(x5)
    x7 = Dense(1, name = 'h4', activation = 'tanh')(x6)
    
    for layer in loaded_model.layers:
        layer.trainable = True

    new_model = Model(loaded_model.input, x7)
    opt = optimizers.Adam(lr=0.0009)
    new_model.compile(loss='mean_squared_error', optimizer= opt)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience = 30)
    history = History()
    os.chdir('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/RuoqiaoChen/DepMap_Public_20q1_Pediatric_Solid_Tumor_Subset/all_model_weights')
    new_model.load_weights(name+'_'+str(count)+'model_weights.h5')#load saved 2000 models
    
    pre = new_model.predict(x_ext)
    pre1 = scaler_y.inverse_transform(pre)

    pre2 = pre1.flatten()
    compare_data = pd.DataFrame((pre2, y_ext))
    compare_data = compare_data.dropna(axis=1)
    pre2 = compare_data.iloc[0,]
    y_ext1 = compare_data.iloc[1,]
    

    mse = mean_squared_error(y_ext1,pre2)
    pearsonr = scipy.stats.pearsonr(y_ext1, pre2)
    spearmanr = scipy.stats.spearmanr(y_ext1, pre2)
    rmse = sqrt(mean_squared_error(y_ext1, pre2))
    r_square = r2_score (y_ext1, pre2)
    Pearsonr = float(pearsonr[0])
    Pearsonr_pvalue = float(pearsonr[1])
    Spearmanr = float(spearmanr[0])
    Spearmanr_pvalue = float(spearmanr[1])
    total_1 = [rmse, mse, Pearsonr, Pearsonr_pvalue,  Spearmanr, Spearmanr_pvalue,  r_square]

    
    return [pre1, total_1]

methods = ['transfer']
y_ext = gene_effect_true[name]


kf = KFold(n_splits=5)
for method in methods:
   
    if method == 'transfer':
       
       X = exp_target.values
       Y = effect_y[name].values
       out=[]
       out1 = []
       count = 0
       for train_index, test_index in kf.split(X):
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = Y[train_index], Y[test_index]
           [pre1, total_1] = transfer_evaluate_model(X_train, y_train, X_test, y_test, new_input, y_ext)
           out.append(list(pre1))
           out1.append(total_1)
           count = count+1
       
       res = np.hstack((out[0], out[1], out[2], out[3], out[4]))
       p_avg = pd.DataFrame({name : np.mean(res, axis=1)})
       output_T = p_avg.T

       
       os.chdir('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/RuoqiaoChen/DepMap_Public_20q1_Pediatric_Solid_Tumor_Subset/new_prediction')

       output_T.to_csv(name + '.csv', header=None)
       res1 = np.array(out1)


       
       res1 = np.array(out1)
       p_avg1 = pd.DataFrame(list(np.mean(res1, axis=0)))
       p_std1 = pd.DataFrame(list(np.std(res1, axis=0)))
       p_avg1.to_csv(name + 'avg_compare_with_true.csv', header=None)
       p_std1.to_csv(name + 'std_compare_with_true.csv', header=None)


  
    
    

















