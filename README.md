# AbClass-Classifier

This repository contains implementations of multiple machine learning and deep learning algorithms for antibody classification models.


## Requirements
- Linux environment
- Python 3.7
- numpy
- pandas
- scikit-learn (sklearn)
- biopython
- matplotlib
- shap


## Algorithms Included
We used the following algorithms to build different classifiers:
- **AdaBoost**
- **CatBoost**
- **LightGBM**
- **Random Forest**
- **Stacking (Ensemble)**
- **Transformer-based Deep Learning Model**
- **XGBoost**


## Workflow for Each Classifier
For all classification algorithms in this project, the following workflow steps are consistently applied:
1. **Data Preprocessing**
2. **Model Construction**
3. **Model Training**
4. **Testing and Evaluation**


## CD-HIT Guide
### 1. Introduction
CD-HIT is a tool for sequence clustering and redundancy reduction. By setting a sequence identity threshold, it clusters highly similar sequences and retains the longest (or most representative) sequence from each cluster, generating a non-redundant dataset.  

In this project, CD-HIT reduces redundancy of raw sequences in `data/raw/`, with the output `data/processed/non_redundant.fasta` used for downstream feature extraction and model training.  

For full details: [CD-HIT Official GitHub](https://github.com/weizhongli/cdhit)


### 2. Installation
#### Option 1: Ubuntu/Debian (Package Manager)
```bash
sudo apt update && sudo apt install cd-hit
```

#### Option 2: All Linux (From Source)
```bash
wget https://github.com/weizhongli/cdhit/archive/refs/tags/V4.8.1.tar.gz
tar -zxvf V4.8.1.tar.gz && cd cdhit-4.8.1 && make
sudo cp cd-hit /usr/local/bin/  # Optional: Add to system PATH
```


### 3. Run Command
Replace `data/raw/your_raw_sequences.fasta` with your raw sequence path:
```bash
cd-hit -i data/raw/your_raw_sequences.fasta \
       -o data/processed/non_redundant.fasta \
       -c 0.4 \          # ≥40% sequence identity threshold
       -n 2 \            # Optimal k-mer length for -c=0.4
       -d 0 \            # Preserve full sequence names
       -M 16000 \        # Memory limit (16GB)
       -T 8              # Use 8 CPU threads
```


### 4. Key Parameters Explained
- `-i`: Input raw sequence file (default: `data/raw/your_raw_sequences.fasta`)
- `-o`: Output prefix (generates `non_redundant.fasta` and cluster details file)
- `-c 0.4`: Cluster sequences with ≥40% identity
- `-n 2`: Optimal k-mer length for `-c=0.4` (ensures clustering accuracy)
- `-d 0`: Preserve full sequence names (no truncation)
- `-M`: Memory limit (MB, adjust by system)
- `-T`: Number of threads (max ≤ CPU core count)


### 5. Output Files
- `data/processed/non_redundant.fasta`: Non-redundant sequences (for downstream use)
- `data/processed/non_redundant.fasta.clstr`: Cluster details (optional verification)


### References
- CD-HIT Official GitHub: [https://github.com/weizhongli/cdhit](https://github.com/weizhongli/cdhit)
- Li W, Godzik A. CD-HIT: a fast program for clustering and comparing large sets of protein or nucleotide sequences. *Bioinformatics*. 2006.


## Feature Guide
This guide details extraction of 3 key features (AAC-PSSM, PseAAC, CTDC) for antibody sequences, using preprocessed `cleaned_sequences.fasta`.


### Step 1: Extract Each Feature Type
| Feature Type | Web Server URL                | Parameter Settings (Manuscript-Aligned) | Output File          |
|--------------|--------------------------------|------------------------------------------|----------------------|
| AAC-PSSM     | https://possum.erc.monash.edu/ | Iterations=60, E-value=0.001, Database=UniRef50 | aac_pssm.csv         |
| PseAAC       | http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/ | PseAA mode=Type 1; Amino acid character=Hydrophobicity, Hydrophilicity, Mass, pK1 (alpha-COOH), pK2 (NH3), pI (25°C); Weight=0.05; Lambda=2 | pse_aac.txt (convert to CSV) |
| CTDC         | https://ifeature.erc.monash.edu/ | Descriptor=CTDC | ctdc.csv |

#### Notes:
- Convert `pse_aac.txt` to CSV (rename to `pse_aac.csv`).
- Save all outputs to `data/processed/`.


### Step 2: Merge Features
Run the script to combine 3 feature files into one matrix:
```bash
python src/feature_extraction/merge_features.py \
       --aac_pssm data/processed/aac_pssm.csv \
       --pse_aac data/processed/pse_aac.csv \
       --ctdc data/processed/ctdc.csv \
       --output data/processed/merged_features.csv
```


## Model Details
### 1. AdaBoost
#### Core Initial Hyperparameters
- Base estimator: Decision Tree (`max_depth=6`)
- Number of estimators: `140`
- Random state: `42`

#### Hyperparameter Search Ranges
| Hyperparameter               | Range          |
|------------------------------|----------------|
| `base_estimator__max_depth`  | [3, 10]        |
| `n_estimators`               | [50, 200]      |
| `learning_rate`              | [0.01, 0.3]    |
| `algorithm`                  | ['SAMME', 'SAMME.R'] |


### 2. CatBoost
#### Core Initial Hyperparameters
- Iterations: `1000`
- Tree depth: `5`
- Learning rate: `0.1`
- Loss function: `MultiClass`
- Random state: `42`

#### Hyperparameter Search Ranges
| Hyperparameter               | Range          |
|------------------------------|----------------|
| `depth`                      | [3, 10]        |
| `learning_rate`              | [0.001, 0.3]   |
| `iterations`                 | [500, 2000]    |
| `l2_leaf_reg`                | [0, 10]        |
| `subsample`                  | [0.6, 1.0]     |


### 3. LightGBM
#### Core Initial Hyperparameters
- Boosting type: `gbdt`
- Number of leaves: `31`
- Learning rate: `0.1`
- Number of estimators: `500`
- Feature fraction: `0.9`
- Bagging fraction: `0.8`
- Bagging frequency: `5`
- Objective: `binary` (use `multiclass` for >2 classes)
- Random state: `42`

#### Hyperparameter Search Ranges
| Hyperparameter               | Range          |
|------------------------------|----------------|
| `num_leaves`                 | [20, 150]      |
| `max_depth`                  | [3, 10]        |
| `learning_rate`              | [0.001, 0.3]   |
| `n_estimators`               | [200, 1000]    |
| `min_child_samples`          | [5, 100]       |


### 4. Random Forest
#### Core Initial Hyperparameters
- Number of estimators: `165`
- Tree depth: `5`
- Minimum samples per leaf: `5`
- OOB score: `True`
- Class weight: `balanced`
- Random state: `42`

#### Hyperparameter Search Ranges
| Hyperparameter               | Range          |
|------------------------------|----------------|
| `n_estimators`               | [100, 300]     |
| `max_depth`                  | [3, 15]        |
| `min_samples_leaf`           | [1, 20]        |
| `max_features`               | ['sqrt', 'log2', 0.5, 0.7, 1.0] |
| `class_weight`               | ['balanced', 'balanced_subsample', None] |


### 5. XGBoost
#### Core Initial Hyperparameters
- Tree depth: `3`
- Learning rate: `0.1`
- Objective: `multi:softmax` (6 classes)
- Subsample: `0.8`
- Colsample by tree: `0.8`
- Eval metric: `mlogloss`
- Random state: `42`

#### Hyperparameter Search Ranges
| Hyperparameter               | Range          |
|------------------------------|----------------|
| `max_depth`                  | [3, 10]        |
| `learning_rate`              | [0.001, 0.3]   |
| `n_estimators`               | [100, 1000]    |
| `gamma`                      | [0, 5]         |
| `reg_alpha`                  | [0, 5]         |
| `reg_lambda`                 | [0, 5]         |


### 6. Stacking Ensemble
#### Architecture
| Layer          | Model Name               | Core Initial Hyperparameters |
|----------------|--------------------------|-------------------------------|
| Base (1st)     | Logistic Regression      | `max_iter=3000`, `C=0.1`, `random_state=1412` |
| Base (1st)     | Random Forest            | `n_estimators=165`, `max_depth=4`, `min_samples_leaf=4`, `random_state=1412` |
| Base (1st)     | SVM                      | `C=10` |
| Base (1st)     | KNN                      | `n_neighbors=10` |
| Final (2nd)    | Random Forest            | `n_estimators=100`, `min_impurity_decrease=0.0025`, `random_state=420` |


### 7. Transformer-based Deep Learning Model
#### Core Initial Hyperparameters
- Number of encoder layers: `6`
- Embedding dimension: `81` (matches input feature count)
- Feedforward dimension: `256` (2-4× embed_dim)
- Number of attention heads: `8` (embed_dim divisible by heads)
- L2 regularization: `0.56`
- Learning rate: `0.001`
- Dropout rate: `0.1`
- Number of classes: `5`

