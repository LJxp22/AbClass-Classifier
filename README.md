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
CD-HIT is a tool for sequence clustering and redundancy reduction. By setting a sequence identity threshold, it clusters highly similar sequences and retains the longest (or most representative) sequence from each cluster, thereby generating a non-redundant dataset. 

In this project, CD-HIT is used to reduce redundancy of raw sequences in the `data/raw/` directory. The output file `data/processed/non_redundant.fasta` provides high-quality data for subsequent feature extraction and model training.

For full details, see: [CD-HIT Official GitHub](https://github.com/weizhongli/cdhit)


### 2. Installation
#### Option 1: Installation via Package Manager (Recommended for Ubuntu/Debian)
```bash
sudo apt update
sudo apt install cd-hit
```

#### Option 2: From Source (All Linux Distributions)
```bash
# Download source code
wget https://github.com/weizhongli/cdhit/archive/refs/tags/V4.8.1.tar.gz

# Extract archive
tar -zxvf V4.8.1.tar.gz

# Enter source directory
cd cdhit-4.8.1

# Compile
make

# Optional: Add to system PATH for global access
sudo cp cd-hit /usr/local/bin/
```


### 3. Command (Run Directly in Linux Terminal)
Replace `[path/to/raw_sequences.fasta]` with the full/relative path to your raw sequence file, and `[path/to/output]` with your desired output directory:
```bash
cd-hit -i data/raw/your_raw_sequences.fasta \
       -o data/processed/non_redundant.fasta \
       -c 0.4 \          # Sequence identity threshold
       -n 2 \            # k-mer length (optimal for -c=0.4)
       -d 0 \            # Preserve full sequence names
       -M 16000 \        # Memory limit (16000 MB = 16 GB)
       -T 8              # Number of threads (max ≤ CPU core count)
```


### 4. Key Parameters Explained
- `-i`: Input raw sequence file (default path: `data/raw/your_raw_sequences.fasta`)
- `-o`: Output prefix (generates `non_redundant.fasta` and a cluster details file)
- `-c 0.4`: Clusters sequences with ≥40% identity (defines the redundancy threshold)
- `-n 2`: Optimal k-mer length for `-c=0.4` (ensures clustering accuracy)
- `-d 0`: Preserve full sequence names in the output file (no truncation)
- `-M`: Memory limit in MB (adjust based on your system's available memory)
- `-T`: Number of threads for parallel processing (speed up clustering)


### 5. Output Files
- `data/processed/non_redundant.fasta`: Non-redundant sequence set (used for downstream analysis)
- `data/processed/non_redundant.fasta.clstr`: Cluster details (optional, for verifying redundancy reduction)


### References
- CD-HIT Official GitHub: [https://github.com/weizhongli/cdhit](https://github.com/weizhongli/cdhit)
- Li W, Godzik A. CD-HIT: a fast program for clustering and comparing large sets of protein or nucleotide sequences. *Bioinformatics*. 2006.


## Feature Guide
This guide details the steps to extract three key feature types (AAC-PSSM, PseAAC, CTDC) using dedicated web servers, consistent with the manuscript **"Computational Models for Predicting Antibody Specificity Using Heavy Chain Features"**. All inputs use the preprocessed `cleaned_sequences.fasta` file.


### 3.1 Step 1: Extract Each Feature Type
| Feature Type | Web Server URL                | Parameter Settings (Manuscript-Aligned) | Output File          |
|--------------|--------------------------------|------------------------------------------|----------------------|
| AAC-PSSM     | https://possum.erc.monash.edu/ | Iterations=60, E-value=0.001, Database=UniRef50 | aac_pssm.csv         |
| PseAAC       | http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/ | PseAA mode=Type 1; Amino acid character=Hydrophobicity, Hydrophilicity, Mass, pK1 (alpha-COOH), pK2 (NH3), pI (at 25°C); Weight factor=0.05; Lambda parameter=2 | pse_aac.txt          |
| CTDC         | https://ifeature.erc.monash.edu/ | Descriptor=CTDC | ctdc.csv |

#### Notes:
- For PseAAC: After downloading the `pse_aac.txt` file, convert it to CSV format (e.g., `pse_aac.csv`).
- Download all output files as CSV and save them to the `data/processed/` directory.


### 3.2 Step 2: Merge Features into One Matrix
Run the provided Python script to combine the three CSV feature files into a single matrix:
```bash
python src/feature_extraction/merge_features.py \
       --aac_pssm data/processed/aac_pssm.csv \
       --pse_aac data/processed/pse_aac.csv \
       --ctdc data/processed/ctdc.csv \
       --output data/processed/merged_features.csv
```
