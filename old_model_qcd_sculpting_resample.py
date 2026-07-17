import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from train_boosted_bdt_all_years import BDT_VARIABLES


QCD_PATTERNS = [
    "Boost_GGJets*.parquet",
    "Boost_GJet*.parquet",
    "Boost_DDQCDGJets*.parquet",
]
WEIGHT_ALIASES = ["eventWeight", "event_weight", "weight_nominal", "weight_central", "weight"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Old-model QCD sculpting check: resample QCD b-tagging from valid signal, "
            "score one old BDT model, and keep each QCD event at most once."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("output_parquet"))
    parser.add_argument("--training-oof", type=Path, default=Path("boosted_bdt_training_all_years/training_oof_scores.parquet"))
    parser.add_argument("--model", type=Path, default=Path("boosted_bdt_training_all_years/xgb_models/xgb_fold0.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("boosted_bdt_training_all_years/sculpting_old_model_qcd"))
    parser.add_argument("--k", type=int, default=500, help="Number of b-tagging resampling rounds.")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--score-cuts", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--max-qcd-events", type=int, default=None, help="Debug option: limit QCD events.")
    return parser.parse_args()


def common_kinematic_mask(df):
    return (
        (df["fatjet_selected_pt"] > 300)
        & (df["fatjet_selected_tau21"] < 0.75)
        & (df["fatjet_selected_regmass"] > 30)
    )


def signal_btag_pool(training_oof):
    df = pd.read_parquet(
        training_oof,
        columns=["Type", "valid", "b-tagging", "fatjet_selected_pt", "fatjet_selected_tau21", "fatjet_selected_regmass", "eventWeight"],
    )
    mask = (df["Type"] == -1) & (df["valid"] == 1) & (df["b-tagging"] > 0) & common_kinematic_mask(df)
    sig = df.loc[mask].copy()
    if len(sig) == 0:
        raise RuntimeError("No valid signal events found for b-tagging target distribution.")
    return sig["b-tagging"].to_numpy(dtype=np.int16), sig


def qcd_files(input_dir):
    files = []
    for pattern in QCD_PATTERNS:
        files.extend(sorted(input_dir.glob(pattern)))
    return sorted(set(files))


def load_qcd(input_dir, max_qcd_events=None):
    files = qcd_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No QCD parquet files found under {input_dir}")

    needed = sorted(
        set(
            BDT_VARIABLES
            + [
                "mass",
                "year",
                "process",
                "source_file",
                "b-tagging",
                "fatjet_selected_pt",
                "fatjet_selected_tau21",
                "fatjet_selected_regmass",
            ]
        )
    )
    pieces = []
    for path in files:
        df = pd.read_parquet(path)
        weight_col = next((col for col in WEIGHT_ALIASES if col in df.columns), None)
        if weight_col is None:
            raise ValueError(f"{path} has no usable weight column. Tried: {WEIGHT_ALIASES}")
        if weight_col != "eventWeight":
            df["eventWeight"] = df[weight_col]
        missing = [col for col in needed if col not in df.columns and col not in {"year", "process", "source_file"}]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df = df.loc[common_kinematic_mask(df)].copy()
        if len(df) == 0:
            continue
        df["source_file"] = path.name
        df["process"] = "QCD"
        pieces.append(df)

    if not pieces:
        raise RuntimeError("No QCD events survived kinematic cuts.")
    out = pd.concat(pieces, ignore_index=True, sort=False)
    out["_event_id"] = np.arange(len(out), dtype=np.int64)
    if max_qcd_events is not None:
        out = out.iloc[:max_qcd_events].copy()
    return out


def predict_one_model(model, x):
    dmatrix = xgb.DMatrix(x)
    best_iteration = model.attr("best_iteration")
    if best_iteration is not None:
        return model.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1)).astype(np.float32)
    return model.predict(dmatrix).astype(np.float32)


def resample_qcd_btag(qcd, signal_btags, model, k, seed):
    rng = np.random.default_rng(seed)
    n = len(qcd)
    best_score = np.full(n, -np.inf, dtype=np.float32)
    best_btag = np.full(n, -1, dtype=np.int16)
    best_resample_id = np.full(n, -1, dtype=np.int32)
    missing = [col for col in BDT_VARIABLES if col not in qcd.columns]
    if missing:
        raise ValueError(f"Missing BDT variables: {missing}")
    base_x = qcd[BDT_VARIABLES].to_numpy(dtype=np.float32)
    btag_idx = BDT_VARIABLES.index("b-tagging")

    for resample_id in range(k):
        sampled_btag = rng.choice(signal_btags, size=n, replace=True)
        x = base_x.copy()
        x[:, btag_idx] = sampled_btag
        score = predict_one_model(model, x)
        update = score > best_score
        if update.any():
            best_score[update] = score[update]
            best_btag[update] = sampled_btag[update]
            best_resample_id[update] = resample_id
        if (resample_id + 1) % 50 == 0 or resample_id == 0:
            print(
                f"resample {resample_id + 1}/{k}: events with saved max score = {int(np.isfinite(best_score).sum())}",
                flush=True,
            )

    keep = np.isfinite(best_score)
    if not keep.any():
        return pd.DataFrame(columns=list(qcd.columns) + ["bdt_score", "resample_id"])

    out = qcd.loc[keep].copy()
    out["b-tagging"] = best_btag[keep]
    out["bdt_score"] = best_score[keep]
    out["resample_id"] = best_resample_id[keep]
    out = out.sort_values("bdt_score", ascending=False).reset_index(drop=True)
    return out


def make_yield_table(df, score_cuts):
    rows = []
    for cut in score_cuts:
        sel = df[df["bdt_score"] > cut]
        sideband = sel[
            ((sel["mass"] >= 100) & (sel["mass"] <= 120))
            | ((sel["mass"] >= 130) & (sel["mass"] <= 180))
        ]
        sr = sel[(sel["mass"] >= 120) & (sel["mass"] <= 130)]
        rows.append(
            {
                "score_cut": cut,
                "n_events": len(sel),
                "eventWeight_sum": sel["eventWeight"].sum(),
                "sideband_n": len(sideband),
                "sideband_eventWeight_sum": sideband["eventWeight"].sum(),
                "sr_120_130_n": len(sr),
                "sr_120_130_eventWeight_sum": sr["eventWeight"].sum(),
            }
        )
    return pd.DataFrame(rows)


def plot_mass(df, score_cuts, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(100, 180, 17)
    for cut in score_cuts:
        sel = df[df["bdt_score"] > cut]
        ax.hist(
            sel["mass"],
            bins=bins,
            range=(100, 180),
            weights=sel["eventWeight"],
            histtype="step",
            linewidth=2,
            label=f"score > {cut:g}",
        )
    ax.set_xlabel("mass")
    ax.set_ylabel("Events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_cuts = [float(x) for x in args.score_cuts.split(",") if x.strip()]

    signal_btags, signal = signal_btag_pool(args.training_oof)
    qcd = load_qcd(args.input_dir, max_qcd_events=args.max_qcd_events)

    model = xgb.Booster()
    model.load_model(args.model)

    print(f"signal valid b-tag pool size: {len(signal_btags)}")
    print("signal b-tagging target distribution:")
    print(signal["b-tagging"].value_counts(normalize=True).sort_index().to_string())
    print(f"qcd events after kinematic cuts, before b-tag cut: {len(qcd)}")
    print("qcd original b-tagging distribution:")
    print(qcd["b-tagging"].value_counts(normalize=True).sort_index().to_string())

    resampled = resample_qcd_btag(
        qcd=qcd,
        signal_btags=signal_btags,
        model=model,
        k=args.k,
        seed=args.seed,
    )

    out_parquet = args.output_dir / "qcd_resampled_old_model.parquet"
    resampled.to_parquet(out_parquet, index=False)

    yield_table = make_yield_table(resampled, score_cuts)
    yield_path = args.output_dir / "qcd_resampled_yields.csv"
    yield_table.to_csv(yield_path, index=False)

    plot_path = args.output_dir / "qcd_resampled_mass.png"
    plot_mass(resampled, score_cuts, plot_path)

    config = pd.DataFrame(
        [
            {
                "model": str(args.model),
                "n_features": len(BDT_VARIABLES),
                "k": args.k,
                "seed": args.seed,
                "n_signal_btag_pool": len(signal_btags),
                "n_qcd_after_kinematic": len(qcd),
                "n_resampled_unique_events": len(resampled),
            }
        ]
    )
    config_path = args.output_dir / "qcd_resampled_config.csv"
    config.to_csv(config_path, index=False)

    print(f"saved resampled events: {out_parquet}")
    print(f"saved yields: {yield_path}")
    print(f"saved mass plot: {plot_path}")
    print(yield_table.to_string(index=False))


if __name__ == "__main__":
    main()
