# Boosted HHbbgg BDT Pipeline

This repository contains the boosted BDT workflow used in the HH to bbgg analysis.
It includes scripts for preparing boosted data/MC samples, training the old boosted
BDT model, evaluating model performance, and checking possible mass sculpting.

Large analysis artifacts are intentionally not tracked in git, including parquet
outputs, pkl inputs, trained XGBoost model files, and local data directories.

## 1. Preprocess Data and MC

MC preprocessing:

```bash
python BDT_preprocess.py
```

This processes `Boost_*.pkl` files and writes parquet outputs under:

```text
output_parquet/
```

Data preprocessing for Run 2, 2024, and 2025:

```bash
python process_DataV6_boost_preselection.py
```

Data preprocessing for 2022 and 2023:

```bash
python process_Data2223_boost_preselection.py
```

Both data scripts write boosted selected samples under:

```text
data/
```

The common boosted selection is:

```text
fatjet_selected_pt > 300
fatjet_selected_tau21 < 0.75
fatjet_selected_regmass > 30
```

For data scoring and final selected data outputs, the common cut also requires:

```text
b-tagging > 0
```

For 2022/2023 data, `b-tagging` is based on `particleNet_XbbVsQCD`.
For later data processed with DataV6, `b-tagging` is based on `globalParT3_XbbVsQCD`.

## 2. Train the Model

The old boosted BDT model is trained with:

```bash
python train_boosted_bdt_all_years.py
```

The training script reads MC parquet files from:

```text
output_parquet/
```

and writes the OOF score output and XGBoost fold models under:

```text
boosted_bdt_training_all_years/
```

The training feature list is defined in:

```python
BDT_preprocess.features
```

## 3. Model Performance

The notebook:

```text
boosted_bdt_training_all_years/Signal_vs_Bkg_AUROC.ipynb
```

contains the main model-performance checks, including:

```text
Signal vs background score distributions
AUROC / ROC checks
Yield scans in mass windows
```

Data scoring with the trained old model can be run with:

```bash
python score_data_mc_with_trained_bdt.py
```

This saves data events with:

```text
bdt_score_mean5 > 0.6
```

to:

```text
boosted_bdt_training_all_years/data_score_gt_0p6.parquet
```

## 4. Sculpting Check

The QCD mass-sculpting check is implemented in:

```text
old_model_qcd_sculpting_resample.py
```

It uses one old-model fold and performs repeated QCD `b-tagging` resampling.
The QCD `b-tagging` value is resampled from the valid signal `b-tagging`
distribution, while each original QCD event is kept only once with its maximum
score across resampling trials.

Run the sculpting check with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python old_model_qcd_sculpting_resample.py --k 500
```

The sculpting outputs are written to:

```text
boosted_bdt_training_all_years/sculpting_old_model_qcd/
```

including:

```text
qcd_resampled_old_model.parquet
qcd_resampled_yields.csv
qcd_resampled_mass.png
qcd_resampled_config.csv
```
