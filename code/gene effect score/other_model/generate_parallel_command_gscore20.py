import pandas as pd
import numpy as np
import csv



gscore_20 = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/output/gscore/other_models/LASSO.csv')
gscore_20 = gscore_20['Unnamed: 0']

j = 0
for i in gscore_20:
    print("python gscore_seperate_prediction_parallel_kfold.py %s > %d.out" % (i,j) )
    j = j + 1


