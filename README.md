# Modular multilabel classifier

Use the existing environment and install the small dependency set:

```bash
conda activate bioinformatics
pip install -r requirements.txt
```

1. Put CSV, XLSX, or Parquet files in `data/raw/`, then run `notebooks/00_data_preprocessing.ipynb`.
2. Set the feature, label, model, and training values in `configs/framework.yaml`.
3. Run `notebooks/01_manual_training.ipynb` to inspect and train the model epoch by epoch.
4. Find the trained model and run outputs in `artifacts/<run_id>/`.
