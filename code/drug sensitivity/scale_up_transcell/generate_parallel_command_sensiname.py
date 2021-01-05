import pandas as pd
import numpy as np
import csv



sensi_name = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/code/sensi/parallel/scaleup_ks5000/sensi_name.csv')
sensi_name = sensi_name['sensi_name']

j = 0

for i in sensi_name:
    print("python sensi_kfold_average_ks5000_parallel_scaleup.py %s > %d.out" % (i,j) )
    j = j + 1


