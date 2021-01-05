import pandas as pd
import numpy as np
import csv



cn_20 = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/output/cnv/other_models/LASSO.csv')
cn_20 = cn_20['Unnamed: 0']

j = 0
for i in cn_20:
    print("python CNV_seperate_prediction_kfold.py %s > %d.out" % (i,j) )
    j = j + 1


