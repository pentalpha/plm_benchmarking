# Protein Dimension DB: Source Code for PLM Benchmarkings

## Requirements

- Cuda capable device
- Minimum 16GB RAM
- Local copy of PDDB Release 2

Packages:

```sh
conda env create -n pyboost --file pyboost_cuda13.yaml
```

## Pipeline

```sh
conda activate pyboost
ln -s <ABS_PATH_TO_PDDB_RELEASE_2> input_data
python bin/sample_metaparameters.py
source test_fuzzy.sh
```