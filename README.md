## In silico expanding of molecular measures from gene expressions through transfer learning 

Personalized cancer therapy drives the emergence of cell lines cultured from individual tumor tissues. Molecular characterization of these cells could gain a better understanding of individual tumors and provide rationale therapeutic options. Gene expression profiling of cell lines becomes routine today; however, obtaining comprehensive molecular characterization and cellular responses for a given cell line is not trivial when resources are very limited. Here, based on the Cancer Dependency Map project (DepMap), we develop TransCell, a deep transfer learning framework that utilizes the knowledge derived from pan-cancer tumor samples to make predictions of mutation, copy number variation (CNV), protein expression, metabolite abundance, gene effect score, and drug sensitivity from gene expression data. In addition to TransCell, state-of-the-art machine learning methods, including two deep neural network architectures, are explored. The results show that TransCell has the best performance in the prediction of complicated tasks: metabolite (Spearman: 0.744 ± 0.044), gene effect score (Spearman: 0.689 ± 0.040), and drug sensitivity (Spearman: 0.653 ± 0.036), and has comparable performance in the prediction of easy task: mutation (AUC mean: 0.863 ± 0.046), CNV (Spearman: 0.806 ± 0.061) and protein expression (Spearman: 0.744 ± 0.103). Of note, TransCell improved the correlation over 50% in drug sensitivity prediction compared to state-of-the-art and published models. This considerable performance of TransCell suggests the potential of in-silico expanding of measures from the easily accessible gene expression. The code of TransCell is available for predicting measures of a given gene expression profile.

### Repository Structure
#### code
All the python scripts are run under the following environments:
- Pyhton version: 3.5.6
- tensorflow version: 1.10.0
- GNU Parallel

Take metabolite prediction as an example.
Commands to run scripts:
```python
python generate_parallel_command_meta20.py > command.txt
```
```python
nohup cat command.txt | parallel --joblog out.log -j5 &
```
```python
cat *.out | grep 'avg_transfer_meta' > avg_transfer_meta_ks5000.csv

```

#### Further application
We could run the following commands to predict the values of metabolte, gene effect score, and drug sensitivity for new cell lines by TransCell.

- Note. The "feature_name" for each individual type could be found in name_meta.csv / name_gscore.csv / name_sensi.csv 

Metabolite:
```python
python further_application_metabolite.py --feature_name metabolite_2-aminoadipate --new_cell_line_name abc --dataset exprs_newCellLine_meta
```

Gene effect score:
```python
python further_application_gene_effect_score.py --feature_name gene_effect_CLDN24 --new_cell_line_name def --dataset exprs_newCellLine_gscore
```

Drug sensitivity:
``` python
python further_application_drug_sensitivity.py --feature_name sensitivity_BRD-A00077618-236-07-6::2.5::HTS --new_cell_line_name ghi --dataset exprs_newCellLine_sensi
```

#### output
The prediction results for each individual type of feature could be found under the folders.

