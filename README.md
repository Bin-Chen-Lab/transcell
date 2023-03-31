## In silico expanding of molecular measures from gene expressions through transfer learning 

Gene expression profiling of new or modified cell lines becomes routine today; however, obtaining comprehensive molecular characterization and cellular responses for a variety of cell lines, including those derived from underrepresented groups, is not trivial when resources are minimal. Using gene expression to predict other measurements has been actively explored; however, systematic investigation of its predictive power in various measurements has not been well studied. We evaluate commonly used machine learning methods and present TransCell, a two-step deep transfer learning framework that utilizes the knowledge derived from pan-cancer tumor samples to predict molecular features and responses. Among these models, TransCell has the best performance in predicting metabolite, gene effect score (or genetic dependency), and drug sensitivity, and has comparable performance in predicting mutation, copy number variation, and protein expression. Notably, TransCell improved the performance by over 50% in drug sensitivity prediction and achieved a correlation of 0.7 in gene effect score prediction. Furthermore, predicted drug sensitivities revealed potential repurposing candidates for new 100 pediatric cancer cell lines, and predicted gene effect scores reflected BRAF resistance in melanoma cell lines. Together, we investigate the predictive power of gene expression in six molecular measurement types and develop a web portal (http://apps.octad.org/transcell/) that enables the prediction of 352,000 genomic and cellular response features solely from gene expression profiles.

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

