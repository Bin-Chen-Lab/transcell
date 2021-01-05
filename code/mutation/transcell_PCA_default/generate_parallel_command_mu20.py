import pandas as pd
import numpy as np
import csv



mu_20 = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/output/mutation/other_models/LOGISTIC_LASSO.csv')
mu_20 = mu_20['Unnamed: 0']

j = 0
for i in mu_20:
    print("python mu_kfold_average_ks5000_parallel.py %s > %d.out" % (i,j) )
    j = j + 1


