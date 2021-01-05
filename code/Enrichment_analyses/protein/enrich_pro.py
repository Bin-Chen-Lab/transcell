#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:36:09 2020

@author: shanjuyeh
"""

import pandas as pd
import numpy as np
#import statistics 


pro_data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/protein/scaleup_elastic/avg_elastic_pro_kfold_scaleup.csv')
meta = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/data/CCLE_RPPA_Ab_info_20181226.csv')

meta = meta[['Antibody_Name', 'Target_Genes']]
pro = pro_data[['Unnamed: 0', 'RMSE_avg']]

tmp = pro['Unnamed: 0']
name=[]
for i in range(len(tmp)):
    tp = tmp[i].split('protein_')[1]
    name.append(tp)

name_df = pd.DataFrame({'name':name})
pro_new = pd.concat([name_df, pro], axis=1)
pro_new = pro_new.drop(columns = ['Unnamed: 0'])
pro_new = pro_new.rename(columns={'name':'Antibody_Name'})

pro_new_use = pd.merge(pro_new, meta, on=['Antibody_Name'])
#pro_rmse_median = statistics.median(list(pro_new_use['RMSE_avg']))

Q1 = np.percentile(list(pro_new_use['RMSE_avg']), 25, interpolation = 'midpoint')
Q3 = np.percentile(list(pro_new_use['RMSE_avg']), 75, interpolation = 'midpoint')



chose_q1 = pro_new_use[pro_new_use['RMSE_avg'] < Q1]
chose_q3 = pro_new_use[pro_new_use['RMSE_avg'] > Q3]

### additional analyses check subunit
sort_q1  = chose_q1.sort_values(by=['RMSE_avg'])
sort_q3 = chose_q3.sort_values(by=['RMSE_avg'])
######


gene_q1 = chose_q1['Target_Genes']
gene_q1 = gene_q1.reset_index(drop=True)

## save name Q1
symbol=[]
for i in range(len(gene_q1)):
    sym = gene_q1[i].split(' ')
    flag = len(sym)
    if flag > 1:
        for j in range(flag):
            tmp = pd.Series(sym[j]).tolist()[0]
            symbol.append(tmp)
    symbol.append(sym[0])
    
symbol_df = pd.DataFrame({'symbol':symbol})
symbol_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/protein/pro_rmse_smallerthan_Q1.csv')      


### save name Q3
gene_q3 = chose_q3['Target_Genes']
gene_q3 = gene_q3.reset_index(drop=True)

symbol=[]
for i in range(len(gene_q3)):
    sym = gene_q3[i].split(' ')
    flag = len(sym)
    if flag > 1:
        for j in range(flag):
            tmp = pd.Series(sym[j]).tolist()[0]
            symbol.append(tmp)
    symbol.append(sym[0])
    
symbol_df = pd.DataFrame({'symbol':symbol})
symbol_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/protein/pro_rmse_largerthan_Q3.csv')      

