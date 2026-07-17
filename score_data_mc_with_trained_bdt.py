import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from BDT_preprocess_ref import FEATURES as REF_BDT_VARIABLES
from train_boosted_bdt_all_years import BDT_VARIABLES, infer_process, infer_year, process_type


DATA_DIR = Path("data")
MC_DIR = Path("output_parquet")
SCORE_CUT = 0.6
MODEL_CONFIGS = {
    "old": {
        "model_pattern": "boosted_bdt_training_all_years/xgb_models/xgb_fold*.json",
        "features": BDT_VARIABLES,
        "output_file": Path("boosted_bdt_training_all_years/data_score_gt_0p6.parquet"),
        "score_column": "bdt_score_mean5",
    },
    "ref": {
        "model_pattern": "boosted_bdt_training_all_years/xgb_ref_models/xgb_ref_fold*.json",
        "features": REF_BDT_VARIABLES,
        "output_file": Path("boosted_bdt_training_all_years/data_ref_score_gt_0p6.parquet"),
        "score_column": "bdt_ref_score_mean5",
    },
}


def apply_common_boosted_cut(df):
    return df[
        (df["b-tagging"] > 0)
        & (df["fatjet_selected_pt"] > 300)
        & (df["fatjet_selected_tau21"] < 0.75)
        & (df["fatjet_selected_regmass"] > 30)
    ].copy()


def load_xgb_models(pattern):
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No models found with pattern: {pattern}")

    models = []
    for path in paths:
        booster = xgb.Booster()
        booster.load_model(path)
        models.append(booster)
    return models


def predict_mean(models, df, bdt_variables):
    missing = [col for col in bdt_variables if col not in df.columns]
    if missing:
        raise ValueError(f"Missing BDT variables: {missing}")

    x = df[bdt_variables].to_numpy(dtype=np.float32)
    dmatrix = xgb.DMatrix(x)
    preds = []
    for model in models:
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            pred = model.predict(dmatrix, iteration_range=(0, best_iteration + 1))
        else:
            pred = model.predict(dmatrix)
        preds.append(pred.astype(np.float32))
    return np.mean(np.stack(preds, axis=0), axis=0)


def parse_data_year(path):
    match = re.search(r"(2016preVFP|2016postVFP|2017|2018|2022|2023|2024|2025)", str(path))
    return match.group(1) if match else "unknown"


def annotate_mc(df, path):
    process = infer_process(path)
    year = infer_year(path)
    df["sample_kind"] = "mc"
    df["process"] = process
    df["year"] = year
    df["source_file"] = Path(path).name
    if process == "signal":
        df["Type"] = -1.0
        df["valid"] = 0 if re.search(r"_batch[45]_", Path(path).name) else 1
    else:
        df["Type"] = process_type(process)
    return df


def annotate_data(df, path):
    df["sample_kind"] = "data"
    df["process"] = "data"
    df["year"] = parse_data_year(path)
    df["source_file"] = Path(path).name
    df["Type"] = np.nan
    return df


def score_files(files, models, annotator, label, bdt_variables, score_column):
    selected = []
    total_rows = 0
    selected_rows = 0

    for path in files:
        df = pd.read_parquet(path)
        total_rows += len(df)
        if len(df) == 0:
            continue

        df = annotator(df, path)
        df = apply_common_boosted_cut(df)
        if len(df) == 0:
            print(f"{label}: {Path(path).name}: 0 after common boosted cut")
            continue
        df[score_column] = predict_mean(models, df, bdt_variables)
        keep = df[score_column] > SCORE_CUT
        out = df.loc[keep].copy()
        selected_rows += len(out)
        if len(out):
            selected.append(out)
        print(f"{label}: {Path(path).name}: {len(df)} -> {len(out)}")

    return selected, total_rows, selected_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Score boosted data with trained old/ref BDT models.")
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="old")
    parser.add_argument("--score-cut", type=float, default=SCORE_CUT)
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    models = load_xgb_models(config["model_pattern"])
    data_files = sorted(DATA_DIR.glob("Run*/*_Boost.parquet"))

    global SCORE_CUT
    SCORE_CUT = args.score_cut
    output_file = Path(args.output_file) if args.output_file else config["output_file"]
    score_column = config["score_column"]

    data_selected, data_total, data_keep = score_files(
        data_files,
        models,
        annotate_data,
        f"data/{args.model}",
        config["features"],
        score_column,
    )

    if not data_selected:
        raise RuntimeError("No events passed score cut.")

    combined = pd.concat(data_selected, ignore_index=True, sort=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_file, index=False)

    summary = pd.DataFrame(
        [
            {
                "sample_kind": "data",
                "model": args.model,
                "n_features": len(config["features"]),
                "score_cut": SCORE_CUT,
                "n_input": data_total,
                "n_score_gt_cut": data_keep,
                "score_column": score_column,
            },
        ]
    )
    summary_path = output_file.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Saved selected events to {output_file}")
    print(f"Saved summary to {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
