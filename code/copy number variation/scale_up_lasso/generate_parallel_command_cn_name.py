import pandas as pd
import numpy as np
import csv


cn_name = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/code/cnv/parallel/scaleup_lasso/cn_name.csv')
cn_name = cn_name['name']

j = 0
for i in cn_name:
    print("python CNV_seperate_prediction_kfold_scaleup.py %s > %d.out" % (i,j) )
    j = j + 1


