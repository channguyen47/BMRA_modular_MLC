# Modular multilabel classifier

Use the existing environment and install the small dependency set:

```bash
conda activate bioinformatics
pip install -r requirements.txt
```

1. Put CSV, XLSX, or Parquet files in `data/raw/`, then run `notebooks/00_data_preprocessing.ipynb`.
2. Set project features and labels in `configs/projects/bmra_student_project.yaml`, and model or training values in `configs/framework.yaml`.
3. Run `notebooks/01_manual_training.ipynb` to inspect and train the model epoch by epoch.
4. Find outputs in `artifacts/<project_basename>/<run_id>/`, then inspect a run with `notebooks/02_classifier_assessment.ipynb`.
