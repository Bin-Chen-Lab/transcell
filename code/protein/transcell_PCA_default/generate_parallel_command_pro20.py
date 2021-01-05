import pandas as pd
import numpy as np
import csv



pro_20 = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/output/protein/other_models/LASSO.csv')
pro_20 = pro_20['Unnamed: 0']

j = 0
for i in pro_20:
    print("python pro_kfold_average_ks5000_parallel.py %s > %d.out" % (i,j) )
    j = j + 1


