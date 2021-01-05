#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:22:27 2020

@author: shanjuyeh
"""

import pandas as pd
import numpy as np
import statistics 


mu_data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/mutation/scaleup_logisticLasso/avg_logisticLasso_mu_kfold_scaleup_NEW.csv')

mu = mu_data[['Unnamed: 0', 'AUC_avg']]

tmp = mu['Unnamed: 0']
name=[]
for i in range(len(tmp)):
    tp = tmp[i].split('mutation_')[1]
    name.append(tp)

name_df = pd.DataFrame({'name':name})
mu_new = pd.concat([name_df, mu], axis=1)
mu_new = mu_new.drop(columns = ['Unnamed: 0'])

auc = mu_new[['AUC_avg']].dropna()

Q1 = np.percentile(list(auc['AUC_avg']), 25)
Q3 = np.percentile(list(auc['AUC_avg']), 75)
#mu_auc_median = statistics.median(list(mu_new['AUC_avg']))
#IQR = Q3 - Q1 

chose_q1 = mu_new[mu_new['AUC_avg'] < Q1]
chose_q3 = mu_new[mu_new['AUC_avg'] > Q3]

gene_q1 = chose_q1['name']
gene_q1 = gene_q1.reset_index(drop=True) ## total 1000
gene_q1_df = pd.DataFrame({'symbol':gene_q1}) ## total 717
gene_q1_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/mutation/mutation_auc_smallerthan_Q1.csv')

gene_q3 = chose_q3['name']
gene_q3 = gene_q3.reset_index(drop=True) ## total 1000
gene_q3_df = pd.DataFrame({'symbol':gene_q3}) ## total 717
gene_q3_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/mutation/mutation_auc_largerthan_Q3.csv')


        