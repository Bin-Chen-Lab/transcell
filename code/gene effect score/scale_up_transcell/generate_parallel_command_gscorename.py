import pandas as pd
import numpy as np
import csv



gscore_name = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/code/gscore/parallel/scale_up/gscore_name.csv')
gscore_name = gscore_name['gscore_name']

j = 0
for i in gscore_name:
    print("python gscore_kfold_average_ks5000_parallel_scaleup.py %s > %d.out" % (i,j) )
    j = j + 1


