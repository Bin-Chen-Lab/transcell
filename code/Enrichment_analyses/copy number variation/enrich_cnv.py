#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:58:04 2020

@author: shanjuyeh
"""

import pandas as pd
import numpy as np
#import statistics 


cnv_data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/CNV/scaleup_lasso/avg_lasso_cn_kfold_scalup.csv')

cnv = cnv_data[['Unnamed: 0', 'RMSE_avg']]

tmp = cnv['Unnamed: 0']
name=[]
for i in range(len(tmp)):
    tp = tmp[i].split('cn_')[1]
    name.append(tp)

name_df = pd.DataFrame({'name':name})
cnv_new = pd.concat([name_df, cnv], axis=1)
cnv_new = cnv_new.drop(columns = ['Unnamed: 0'])


Q1 = np.percentile(list(cnv_new['RMSE_avg']), 25, interpolation = 'midpoint')
Q3 = np.percentile(list(cnv_new['RMSE_avg']), 75, interpolation = 'midpoint')

#cnv_rmse_mean = statistics.mean(list(cnv_new['RMSE_avg']))
#cnv_rmse_median = statistics.median(list(cnv_new['RMSE_avg']))
#chose = cnv_new[cnv_new['RMSE_avg']<cnv_rmse_median]
#chose = cnv_new[cnv_new['RMSE_avg']<cnv_rmse_mean]


chose_q1 = cnv_new[cnv_new['RMSE_avg'] < Q1]
chose_q3 = cnv_new[cnv_new['RMSE_avg'] > Q3]

gene_q1 = chose_q1['name']
gene_q1 = gene_q1.reset_index(drop=True) ## total 500
gene_q1_df = pd.DataFrame({'symbol':gene_q1})
gene_q1_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/cnv/cnv_rmse_smallerthan_Q1.csv')

gene_q3 = chose_q3['name']
gene_q3 = gene_q3.reset_index(drop=True) ## total 500
gene_q3_df = pd.DataFrame({'symbol':gene_q3})
gene_q3_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/cnv/cnv_rmse_largerthan_Q3.csv')




#####===========================================================================
#cnv = cnv_data[['Unnamed: 0', 'Spearmanr_avg']]
#
#tmp = cnv['Unnamed: 0']
#name=[]
#for i in range(len(tmp)):
#    tp = tmp[i].split('cn_')[1]
#    name.append(tp)
#
#name_df = pd.DataFrame({'name':name})
#cnv_new = pd.concat([name_df, cnv], axis=1)
#cnv_new = cnv_new.drop(columns = ['Unnamed: 0'])
#
#
#cnv_sort = cnv_new.sort_values(by=['Spearmanr_avg'], ascending=False)
#cnv_sort = cnv_sort.reset_index(drop=True)
#cnv_easy = cnv_sort['name'][0:500]
#cnv_hard = cnv_new['name'][-500:]
#
#cnv_easy = cnv_easy.reset_index(drop=True)
#cnv_hard = cnv_hard.reset_index(drop=True)
#
#easy_df = pd.DataFrame({'name':cnv_easy})
#hard_df = pd.DataFrame({'name':cnv_hard})
#
#easy_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/cnv/cnv_easy500_spearman.csv')
#hard_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/cnv/cnv_hard500_spearman.csv')


##### Using speraman correlation, check Q1 and Q3
#cnv_data = pd.read_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/output/CNV/scaleup_lasso/avg_lasso_cn_kfold_scalup.csv')
#
#cnv = cnv_data[['Unnamed: 0', 'Spearmanr_avg']]
#
#tmp = cnv['Unnamed: 0']
#name=[]
#for i in range(len(tmp)):
#    tp = tmp[i].split('cn_')[1]
#    name.append(tp)
#
#name_df = pd.DataFrame({'name':name})
#cnv_new = pd.concat([name_df, cnv], axis=1)
#cnv_new = cnv_new.drop(columns = ['Unnamed: 0'])
#
#cnv_sp_median = statistics.median(list(cnv_new['Spearmanr_avg']))
#
##Q1 = np.percentile(list(cnv_new['RMSE_avg']), 25, interpolation = 'midpoint') 
##cnv_rmse_mean = statistics.mean(list(cnv_new['RMSE_avg']))
#
#chose = cnv_new[cnv_new['Spearmanr_avg'] > cnv_sp_median]
##chose = cnv_new[cnv_new['RMSE_avg']<cnv_rmse_mean]
##chose = cnv_new[cnv_new['RMSE_avg']<Q1]
#
#
#gene = chose['name']
#gene = gene.reset_index(drop=True) ## total 1000
#gene_df = pd.DataFrame({'symbol':gene})
#
#gene_df.to_csv('/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/cnv/cnv_larger_median_spearman.csv')
#











        