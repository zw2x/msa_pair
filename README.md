# MsaPair

## Installation
```
git clone https://github.com/zw2x/msa_pair.git
cd msa_pair
pip install -e .
```
### Requirments
Install the latest versions of alphafold and esm.

## Dataset

Download the dataset 
[here](https://www.dropbox.com/s/48jg26m3jlgt1uj/below_medium.tar.gz?dl=0). 
It includes **176** subdirectories; each corresponds to a PDB target. The 
structure of each of the subdirectory looks like this:
```
- A: .a3ms for chain A
- B: .a3ms for chain B
- esm_scores.json: the esm scores for sequences in uniprot.a3m
- esm_pr.json: the paired rows based on "esm_scores.json"
- multimer.npz: the input feature to AlphaFold-Multimer; this is derived from 
    "esm_pr.json"
```

## Multimer-pipeline

### AlphaFold-Multimer default pipeline
```
python scripts/build_alphafold_features.py below_medium
```

### MSA-Transformer ColAttn pipeline
After modifing `esm/model.py` using the `model.py` file provided by Bo Chen, run

```
python scripts/build_colattn_features.py below_medium
```
You can also replace the `compute_scores` function in the script with your own 
scoring function, and then use the `pair_rows` and `process` in the script to 
generate features for AlphaFold-Multimer

### Genetic locus pipeline
This pipeline needs fast internet connections and a lot of memory and disk 
space, because we need to download and process whole genome sequence (WGS) data 
from [ENA](https://www.ebi.ac.uk/ena/browser/).

Download required WGS data from ENA
```
python scripts/export_ena_requests below_medium
```

Pair sequences
```
python scripts/build_ena_features.py below_medium
```

## Run AlphaFold-Multimer
```
python msa_pair/scripts/run_model.py --data-dir database \
    --input-npz multimer.npz --output-dir models
```
This script can run all AlphaFold-Multimer models but need only compile the 
code once.

## Assess
To assess the quality of `prediction.pdb` against the true pdb 
`ground_truth.pdb`, run:
```bash
python scripts/assess_models.py --query prediction.pdb --ground-truth \
    ground_truth.pdb --dst-dir assess_results --tmalign-path ${TMALIGN_BINARY} \
    --dockq-path ${DOCKQ_PY}
```

All outputs will be in `assess_results`. The most important output file is 
`assess.json`, which shows the DockQ scores.
