import pandas as pd
import numpy as np
import csv



sensi_20 = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/output/sensi/other_models/LASSO.csv')
sensi_20 = sensi_20['Unnamed: 0']

j = 0

for i in sensi_20:
    print("python sensi_kfold_average_ks5000_parallel.py %s > %d.out" % (i,j) )
    j = j + 1


