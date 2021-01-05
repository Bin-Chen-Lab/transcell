import pandas as pd
import numpy as np
import csv



meta_20 = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/output/meta/other_models/LASSO.csv')
meta_20 = meta_20['Unnamed: 0']

j = 0
for i in meta_20:
    print("python meta_patientEncoder_ks5000.py %s > %d.out" % (i,j) )
    j = j +1


