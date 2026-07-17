import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from BDT_preprocess import features as BDT_VARIABLES


WEIGHT_ALIASES = ["eventWeight", "event_weight", "weight_nominal", "weight"]
EXTRA_COLUMNS = [
    "mass",
    "erainfo",
    "fatjet_selected_tau_ratio",
    "fatjet_selected_genMatched_Hbb",
    "fatjet_selected_pnetmass",
]

TYPE_LABELS = {
    -1: "Signal",
    0: "GGJ/GJ/DDQCD",
    1: "TTG/TTGG",
    2: "Unmatched signal",
    3: "ggH",
    4: "ttH/bbH",
    5: "VH",
    6: "VBFH",
}

RANDOM_SEED = 42
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.09,
    "gamma": 0.1,
    "seed": RANDOM_SEED,
    "subsample": 0.6,
}


def infer_year(path):
    stem = Path(path).stem
    for token in ["2016pre", "2016post", "2017", "2018", "2223", "2024", "2025"]:
        if token in stem:
            return token
    return "unknown"


def infer_process(path):
    name = Path(path).name
    if name.startswith("Boost_GluGlutoHH_"):
        return "signal"
    if name.startswith("Boost_GluGluHtoGG_"):
        return "ggH"
    if name.startswith("Boost_ttHtoGG_"):
        return "ttH"
    if name.startswith("Boost_bbHtoGG_"):
        return "bbH"
    if name.startswith("Boost_VHtoGG_"):
        return "VH"
    if name.startswith("Boost_VBFHtoGG_"):
        return "VBFH"
    if name.startswith("Boost_GGJets_") or name.startswith("Boost_GJet_") or name.startswith("Boost_DDQCDGJets_"):
        return "QCD"
    if name.startswith("Boost_TTGG_") or name.startswith("Boost_TTG-"):
        return "TTG"
    return None


def process_type(process):
    if process == "QCD":
        return 0.0
    if process == "TTG":
        return 1.0
    if process == "ggH":
        return 3.0
    if process in {"ttH", "bbH"}:
        return 4.0
    if process == "VH":
        return 5.0
    if process == "VBFH":
        return 6.0
    return np.nan


def columns_to_read(path, mass_column):
    schema_cols = set(pq.read_schema(path).names)
    wanted = set(BDT_VARIABLES + EXTRA_COLUMNS + WEIGHT_ALIASES + [mass_column])
    return [col for col in wanted if col in schema_cols]


def choose_weight_column(df):
    for col in WEIGHT_ALIASES:
        if col in df.columns:
            return col
    raise ValueError(f"No weight column found. Tried: {WEIGHT_ALIASES}")


def load_sample(path, mass_column):
    process = infer_process(path)
    if process is None:
        return None

    df = pd.read_parquet(path, columns=columns_to_read(path, mass_column))
    weight_col = choose_weight_column(df)
    if weight_col != "eventWeight":
        df["eventWeight"] = df[weight_col]

    df["process"] = process
    df["year"] = infer_year(path)
    df["source_file"] = Path(path).name

    if process == "signal":
        df["valid"] = 0 if re.search(r"_batch[45]_", Path(path).name) else 1
    else:
        df["Type"] = process_type(process)

    return df


def apply_notebook_cuts(df, mass_column):
    if mass_column not in df.columns:
        raise ValueError(f"Mass cut column '{mass_column}' is missing from dataframe.")
    mask = (
        (df["b-tagging"] > 0)
        & (df["fatjet_selected_tau21"] < 0.75)
        & (df["fatjet_selected_pt"] > 300)
        & (df[mass_column] > 30)
    )
    return df.loc[mask].copy()


def check_required_columns(df, columns, name):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def build_training_frames(input_dir, mass_column):
    files = sorted(Path(input_dir).glob("Boost_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Boost_*.parquet files found in {input_dir}")

    samples = []
    skipped = []
    for path in files:
        df = load_sample(path, mass_column)
        if df is None:
            skipped.append(path.name)
            continue
        samples.append(apply_notebook_cuts(df, mass_column))

    if skipped:
        print("Skipped unrecognized files:", skipped)

    all_samples = pd.concat(samples, ignore_index=True)
    check_required_columns(all_samples, BDT_VARIABLES + ["eventWeight", "mass"], "all samples")

    signal_all = all_samples.loc[all_samples["process"] == "signal"].copy()
    check_required_columns(signal_all, ["fatjet_selected_genMatched_Hbb"], "signal")

    signal = signal_all.loc[signal_all["fatjet_selected_genMatched_Hbb"] == 1].copy()
    fake_signal = signal_all.loc[signal_all["fatjet_selected_genMatched_Hbb"] != 1].copy()

    signal["Type"] = -1.0
    fake_signal["Type"] = 2.0

    ggh_weight_sum = all_samples.loc[all_samples["process"] == "ggH", "eventWeight"].sum()
    fake_weight_sum = fake_signal["eventWeight"].sum()
    if len(fake_signal) > 0 and fake_weight_sum != 0 and ggh_weight_sum != 0:
        fake_signal["eventWeight"] = fake_signal["eventWeight"] * (ggh_weight_sum / fake_weight_sum)

    single_h_processes = ["ggH", "ttH", "bbH", "VH", "VBFH"]
    single_h = all_samples.loc[all_samples["process"].isin(single_h_processes)].copy()
    single_h = pd.concat([single_h, fake_signal], ignore_index=True)

    nonres = all_samples.loc[all_samples["process"].isin(["QCD", "TTG"])].copy()

    # Keep the notebook's Type==4 scaling exactly. In that notebook Type==4 includes ttH and bbH.
    single_h.loc[single_h["Type"] == 4, "eventWeight"] *= 170 / 69

    nonres_weight = nonres["eventWeight"].sum()
    single_h["balanced_Weight"] = 0.5 * single_h["eventWeight"] * (
        nonres_weight / single_h["eventWeight"].sum()
    )
    signal["balanced_Weight"] = signal["eventWeight"] * (nonres_weight / signal["eventWeight"].sum())
    nonres["balanced_Weight"] = 0.5 * nonres["eventWeight"]

    keep_signal = BDT_VARIABLES + [
        "eventWeight",
        "balanced_Weight",
        "erainfo",
        "mass",
        "valid",
        "fatjet_selected_tau_ratio",
        "fatjet_selected_genMatched_Hbb",
        "Type",
        "process",
        "year",
        "source_file",
    ]
    keep_bkg = BDT_VARIABLES + [
        "eventWeight",
        "balanced_Weight",
        "erainfo",
        "Type",
        "mass",
        "process",
        "year",
        "source_file",
    ]

    signal = signal[[col for col in keep_signal if col in signal.columns]].copy()
    single_h = single_h[[col for col in keep_bkg if col in single_h.columns]].copy()
    nonres = nonres[[col for col in keep_bkg if col in nonres.columns]].copy()

    return signal, single_h, nonres


def make_kfold_oof_xgb(
    signal_df,
    single_h_df,
    nonres_df,
    bdt_variables,
    k=5,
    seed=42,
    params=None,
    num_boost_round=1500,
    early_stopping_rounds=50,
    use_abs_weight=True,
):
    if params is None:
        params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.09, "gamma": 0.1, "subsample": 0.6}

    sig = signal_df.copy()
    sig["label"] = 1.0
    if "Type" not in sig.columns:
        sig["Type"] = -1

    bkg = pd.concat([nonres_df.copy(), single_h_df.copy()], ignore_index=True)
    bkg["label"] = 0.0
    if "Type" not in bkg.columns:
        raise ValueError("Background dataframe needs a Type column.")

    df_all = pd.concat([sig, bkg], ignore_index=True)
    check_required_columns(df_all, bdt_variables + ["balanced_Weight", "label", "Type"], "training frame")

    weights = df_all["balanced_Weight"].to_numpy()
    if use_abs_weight:
        weights = np.abs(weights)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    x = df_all[bdt_variables].to_numpy(dtype=np.float32)
    y = df_all["label"].to_numpy(dtype=np.float32)
    type_arr = df_all["Type"].fillna(-1).astype(int).to_numpy()
    strata = (y.astype(int) * 100 + (type_arr + 50)).astype(int)

    strata_counts = Counter(strata)
    min_stratum = min(strata_counts.values())
    if min_stratum < k:
        raise ValueError(
            f"Cannot run {k}-fold split: smallest (label, Type) stratum has {min_stratum} rows."
        )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof_score = np.full(len(df_all), np.nan, dtype=np.float32)
    fold_id = np.full(len(df_all), -1, dtype=int)
    models = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, strata)):
        fold_id[valid_idx] = fold

        dtrain = xgb.DMatrix(x[train_idx], label=y[train_idx], weight=weights[train_idx])
        dvalid = xgb.DMatrix(x[valid_idx], label=y[valid_idx], weight=weights[valid_idx])

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        if getattr(booster, "best_iteration", None) is not None:
            pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        else:
            pred = booster.predict(dvalid)

        oof_score[valid_idx] = pred.astype(np.float32)
        models.append(booster)
        print(
            f"[fold {fold}] train={len(train_idx)} valid={len(valid_idx)} "
            f"best_iter={getattr(booster, 'best_iteration', None)}"
        )

    df_all = df_all.copy()
    df_all["fold"] = fold_id
    df_all["oof_score"] = oof_score

    n_nan = np.isnan(df_all["oof_score"].to_numpy()).sum()
    if n_nan:
        raise RuntimeError(f"OOF score has {n_nan} NaN entries.")

    return df_all, models


def save_outputs(
    df_all,
    models,
    output_dir,
    oof_name="training_oof_scores.parquet",
    summary_name="training_summary.csv",
    model_subdir="xgb_models",
):
    output_dir = Path(output_dir)
    model_dir = output_dir / model_subdir
    model_dir.mkdir(parents=True, exist_ok=True)

    df_all.to_parquet(output_dir / oof_name, index=False)
    for i, booster in enumerate(models):
        booster.save_model(model_dir / f"xgb_fold{i}.json")

    summary = (
        df_all.groupby(["label", "Type", "process", "year"], dropna=False)
        .agg(n=("label", "size"), eventWeight_sum=("eventWeight", "sum"), balancedWeight_sum=("balanced_Weight", "sum"))
        .reset_index()
    )
    summary.to_csv(output_dir / summary_name, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train boosted BDT with BoostBDT_matching.ipynb logic for all years.")
    parser.add_argument("--input-dir", default="output_parquet", help="Directory containing Boost_*.parquet files.")
    parser.add_argument("--output-dir", default="boosted_bdt_training_all_years", help="Directory for models and OOF output.")
    parser.add_argument("--mass-column", default="fatjet_selected_regmass", help="Fatjet mass column used for >30 cut.")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num-boost-round", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Build samples and print counts without training.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal, single_h, nonres = build_training_frames(args.input_dir, args.mass_column)

    print("After notebook-style cuts:")
    print(f"  Signal matched: {len(signal)}")
    print(f"  SingleH + fake signal: {len(single_h)}")
    print(f"  NonRes: {len(nonres)}")
    print("Type labels:", TYPE_LABELS)

    if args.dry_run:
        return

    df_all, models = make_kfold_oof_xgb(
        signal_df=signal,
        single_h_df=single_h,
        nonres_df=nonres,
        bdt_variables=BDT_VARIABLES,
        k=args.kfold,
        seed=args.seed,
        params={**XGB_PARAMS, "seed": args.seed},
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        use_abs_weight=True,
    )

    save_outputs(df_all, models, output_dir)
    print(f"Saved OOF scores and models under {output_dir}")


if __name__ == "__main__":
    main()
