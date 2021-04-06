import pandas as pd
import numpy as np
import csv

gene_effect_name = pd.read_csv('/home/ubuntu/chenlab_deeplearning/chenlab_deeplearning_V2/RuoqiaoChen/DepMap_Public_20q1_Pediatric_Solid_Tumor_Subset/gscore_name.csv')
gene_effect_name = gene_effect_name['gscore_name']

j = 0

for i in gene_effect_name:
    print("gene_effect_new_input.py %s > %d.out" % (i,j) )
    j = j + 1


