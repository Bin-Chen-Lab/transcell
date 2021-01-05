#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:43:06 2020

@author: shanjuyeh
"""

import pandas as pd
import numpy as np
import statistics 

sensi_data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/sensi/transfer_learning/scaleup_KS5000/avg_transfer_sensi_ks5000_scaleup.csv')
meta = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/primary-screen-replicate-collapsed-treatment-info .csv')

sensi = sensi_data[['Unnamed: 0', 'RMSE_avg']]
meta = meta[['column_name', 'target']]

tmp = sensi['Unnamed: 0']
name=[]
for i in range(len(tmp)):
    tp = tmp[i].split('sensitivity_')[1]
    name.append(tp)

name_df = pd.DataFrame({'name':name})
sensi_new = pd.concat([name_df, sensi], axis=1)
sensi_new = sensi_new.drop(columns = ['Unnamed: 0'])
sensi_new = sensi_new.rename(columns={'name':'column_name'})

sensi_new_use = pd.merge(sensi_new, meta, on=['column_name'])

#sensi_rmse_median = statistics.median(list(sensi_new_use['RMSE_avg']))

Q1 = np.percentile(list(sensi_new_use['RMSE_avg']), 25, interpolation = 'midpoint')
Q3 = np.percentile(list(sensi_new_use['RMSE_avg']), 75, interpolation = 'midpoint')
#IQR = Q3 - Q1 

chose_q1 = sensi_new_use[sensi_new_use['RMSE_avg'] < Q1]
chose_q3 = sensi_new_use[sensi_new_use['RMSE_avg'] > Q3]


### outlier
#chose_q1 = sensi_new_use[sensi_new_use['RMSE_avg'] < Q1 - 1.5*IQR]
#chose_q3 = sensi_new_use[sensi_new_use['RMSE_avg'] > Q3 + 1.5*IQR]
#chose = sensi_new_use[sensi_new_use['RMSE_avg'] < sensi_rmse_median]


##### Q1 outliers
gene_q1 = chose_q1[['target']]
gene_q1 = gene_q1.dropna()
gene_q1 = gene_q1.reset_index(drop=True)
gene_series_q1 = gene_q1['target']

## save name
symbol=[]
for i in range(len(gene_series_q1)):
    sym = gene_series_q1[i].split(',')
    flag = len(sym)
    if flag > 1:
        for j in range(flag):
            tmp = pd.Series(sym[j]).tolist()[0]
            symbol.append(tmp)
    symbol.append(sym[0])
    
symbol_df = pd.DataFrame({'symbol':symbol})
symbol_df.symbol = symbol_df.symbol.str.lstrip() ## remove space at the beginning of str
symbol_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/sensi/sensi_rmse_smallerthan_Q1.csv')
        

####  Q3 outliers
gene_q3 = chose_q3[['target']]
gene_q3 = gene_q3.dropna()
gene_q3 = gene_q3.reset_index(drop=True)
gene_series_q3 = gene_q3['target']

## save name
symbol=[]
for i in range(len(gene_series_q3)):
    sym = gene_series_q3[i].split(',')
    flag = len(sym)
    if flag > 1:
        for j in range(flag):
            tmp = pd.Series(sym[j]).tolist()[0]
            symbol.append(tmp)
    symbol.append(sym[0])
    
symbol_df = pd.DataFrame({'symbol':symbol})
symbol_df.symbol = symbol_df.symbol.str.lstrip() ## remove space at the beginning of str
symbol_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/sensi/sensi_rmse_largerthan_Q3.csv')
        