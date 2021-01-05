import pandas as pd
import numpy as np
import csv



meta_name = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/code/meta/parallel/KS5000/scale_up/meta_name.csv')
meta_name = meta_name['meta_name']

j = 0
for i in meta_name:
    print("python meta_kfold_average_ks5000_parallel_scaleup.py %s > %d.out" % (i,j) )
    j = j +1


