import pandas as pd
import numpy as np
import csv


pro_name = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/DL_yeh/GeneExp_prediction/code/protein/parallel/scaleup_elastic/protein_name.csv')
pro_name = pro_name['name']

j = 0
for i in pro_name:
    print("python protein_seperate_prediction_kfold_scaleup.py '%s' > %d.out" % (i,j) )
    j = j + 1


