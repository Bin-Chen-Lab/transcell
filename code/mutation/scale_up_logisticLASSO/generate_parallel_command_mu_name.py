import pandas as pd
import numpy as np
import csv



mu_name = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/code/mutation/parallel/scaleup_logisticLasso/mu_name.csv')
mu_name = mu_name['name']

j = 0
for i in mu_name:
    print("python mu_seperate_prediction_kfold_scaleup.py %s > %d.out" % (i,j) )
    j = j + 1


