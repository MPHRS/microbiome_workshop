# Microbiome analysis workshop - single bioproject

A brief example of microbiome data analysis. For more complex analysis, see the [main repository](https://github.com/MPHRS/GNTK_M).

## Files and why they exist

1. `01_data_overview.ipynb`  
   Loads `metadata.csv`, `tax.csv` and `path.csv`, aligns samples by `run`, checks group counts and matrix sizes.

2. `02_alpha_diversity_and_pca.ipynb`  
   Builds basic exploratory plots: alpha diversity and CLR-PCA.  
   True batch-effect analysis is not included because the workshop dataset has only one bioproject.

3. `03_differential_abundance_maaslin2_template.ipynb`  
   R template. It only shows how to run MaAsLin2 for `K05 vs healthy` and `K02 vs healthy`. 

4. `04_ml_cv.ipynb`  
   It runs stratified cross-validation, reports AUC/MCC/BACC and plots ROC curves for all folds.

5. `simple_utils.py`  
   Small shared code used by the notebooks: loading/alignment, CLR through scikit-bio, alpha diversity through scikit-bio, train/test preparation and logistic regression helpers.

6. `requirements.txt`  
   Python dependencies.

`05_ml_loso.ipynb` was removed. LOSO requires at least two bioprojects. With one bioproject it is methodologically not meaningful.

## Expected Python input files

```text
data/tax.csv
data/path.csv
data/metadata.csv
```

Required metadata columns:

```text
run
bioproject
healthy
K02_caries
K05_gingivitis_periodontitis
```

Change paths only in the first code cell of each notebook.

## Expected MaAsLin2 input files

```text
data/microbiome_383_taxa_with_sampleid.csv
data/has_K05_sampleids.csv
data/has_K02_sampleids.csv
```

Disease status is encoded as:

```text
1 = healthy/control
2 = case
```

