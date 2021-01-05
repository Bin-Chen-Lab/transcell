#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:33:44 2020

@author: shanjuyeh
"""

import pandas as pd
import numpy as np
#import statistics 


gscore_data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/gscore/scaleup_TransCell_2000gscore/avg_tansfer_gscore_ks5000_scaleup2000.csv')

gscore = gscore_data[['Unnamed: 0', 'RMSE_avg']]

tmp = gscore['Unnamed: 0']
name=[]
for i in range(len(tmp)):
    tp = tmp[i].split('gene_effect_')[1]
    name.append(tp)

name_df = pd.DataFrame({'name':name})
gscore_new = pd.concat([name_df, gscore], axis=1)
gscore_new = gscore_new.drop(columns = ['Unnamed: 0'])

#gscore_rmse_median = statistics.median(list(gscore_new['RMSE_avg']))
Q1 = np.percentile(list(gscore_new['RMSE_avg']), 25, interpolation = 'midpoint')
Q3 = np.percentile(list(gscore_new['RMSE_avg']), 75, interpolation = 'midpoint')


chose_q1 = gscore_new[gscore_new['RMSE_avg'] < Q1]
gene_q1 = chose_q1['name']
gene_q1 = gene_q1.reset_index(drop=True) ## total 500
gene_q1_df = pd.DataFrame({'symbol':gene_q1}) ## total 500
gene_q1_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/gscore/gscore_rmse_smallerthan_Q1.csv')

chose_q3 = gscore_new[gscore_new['RMSE_avg'] > Q3]
gene_q3 = chose_q3['name']
gene_q3 = gene_q3.reset_index(drop=True) ## total 500
gene_q3_df = pd.DataFrame({'symbol':gene_q3}) ## total 500
gene_q3_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/gscore/gscore_rmse_largerthan_Q3.csv')
        