import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import vector
from BDT_preprocess_ref import FEATURES as REF_FEATURES
from pandas.errors import PerformanceWarning


DATA_V6_DIR = Path("/home/zhong202/depot/users/zhong202/HHtoBBGG/HDNA_sample/v3_parquet_processing/DataV6")
DEFAULT_OUTPUT_DIR = Path("data")
warnings.simplefilter("ignore", PerformanceWarning)

DERIVED_COLUMNS_FOR_EMPTY_OUTPUT = [
    *REF_FEATURES,
    "fatjet_selected_globalParT3_XbbVsQCD",
    "fatjet_selected_regmass",
    "fatjet_selected_tau21",
    "fatjet_selected_tau32",
    "b-tagging",
    "fatjet_selected_tau2tau1_ratio",
    "fatjet_selected_Xbb_wp2",
    "fatjet_selected_Xbb_wp3",
    "fatjet_selected_Xbb_wp4",
    "fatjet_selected_Xbb_wp5",
    "DeltaR_jg_min",
    "mass_HH",
    "pt_HH",
    "eta_HH",
    "lead_sigmaE_over_E",
    "sublead_sigmaE_over_E",
    "n_fatjets_final",
    "year",
]


def infer_year(path):
    text = str(path)
    for token in ["2016preVFP", "2016postVFP", "2017", "2018", "2024", "2025"]:
        if token in text:
            return token
    return "unknown"


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def safe_divide(num, den):
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=(den != 0))


def add_fatjet_base_variables(df):
    for i in range(1, 5):
        tau1 = df[f"fatjet{i}_tau1"].to_numpy(dtype=float)
        tau2 = df[f"fatjet{i}_tau2"].to_numpy(dtype=float)
        tau3 = df[f"fatjet{i}_tau3"].to_numpy(dtype=float)
        df[f"fatjet{i}_tau_ratio"] = safe_divide(tau2, tau1)
        df[f"fatjet{i}_tau21"] = df[f"fatjet{i}_tau_ratio"]
        df[f"fatjet{i}_tau32"] = safe_divide(tau3, tau2)

        if f"fatjet{i}_globalParT3_Xbb" in df.columns and f"fatjet{i}_globalParT3_QCD" in df.columns:
            xbb = df[f"fatjet{i}_globalParT3_Xbb"].to_numpy(dtype=float)
            qcd = df[f"fatjet{i}_globalParT3_QCD"].to_numpy(dtype=float)
            df[f"fatjet{i}_globalParT3_XbbVsQCD"] = safe_divide(xbb, xbb + qcd)

        if f"fatjet{i}_mass_raw" in df.columns and f"fatjet{i}_globalParT3_massCorrX2p" in df.columns:
            df[f"fatjet{i}_regmass"] = (
                df[f"fatjet{i}_mass_raw"] * df[f"fatjet{i}_globalParT3_massCorrX2p"]
            )


def choose_xbb_columns(df, year):
    xbb_cols = [f"fatjet{i}_globalParT3_XbbVsQCD" for i in range(1, 5)]
    if all(col in df.columns for col in xbb_cols):
        return xbb_cols
    missing = [col for col in xbb_cols if col not in df.columns]
    raise KeyError(f"Cannot find complete globalParT3 Xbb score columns for {year}. Missing: {missing}")


def add_selected_fatjet(df, best_idx):
    prop_dict = {}
    pattern = re.compile(r"fatjet(\d+)_(.+)")
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        jet_index = int(match.group(1))
        prop = match.group(2)
        if 1 <= jet_index <= 4:
            prop_dict.setdefault(prop, []).append((jet_index, col))

    for prop, jets in prop_dict.items():
        jets_sorted = sorted(jets, key=lambda item: item[0])
        if len(jets_sorted) != 4:
            continue
        col_names = [col for _, col in jets_sorted]
        values = df[col_names].to_numpy()
        df[f"fatjet_selected_{prop}"] = np.take_along_axis(
            values, best_idx[:, None], axis=1
        ).ravel()


def add_btagging(df, year):
    xbb = df["fatjet_selected_globalParT3_XbbVsQCD"]
    bins = [
        ((xbb >= 0.25) & (xbb < 0.5), 1),
        ((xbb >= 0.5) & (xbb < 0.8), 2),
        ((xbb >= 0.8) & (xbb < 0.9), 3),
        ((xbb >= 0.9) & (xbb < 0.95), 4),
        (xbb >= 0.95, 5),
    ]

    df["b-tagging"] = 0
    for mask, value in bins:
        df.loc[mask, "b-tagging"] = value

    df["fatjet_selected_Xbb_wp2"] = ((xbb >= 0.5) & (xbb < 0.8)).astype(int)
    df["fatjet_selected_Xbb_wp3"] = ((xbb >= 0.8) & (xbb < 0.9)).astype(int)
    df["fatjet_selected_Xbb_wp4"] = ((xbb >= 0.9) & (xbb < 0.95)).astype(int)
    df["fatjet_selected_Xbb_wp5"] = (xbb >= 0.95).astype(int)


def add_bdt_variables(df):
    df["fatjet_selected_tau2tau1_ratio"] = df["fatjet_selected_tau21"]
    df["lead_sigmaE_over_E"] = df["lead_energyErr"] / (df["lead_pt"] * np.cosh(df["lead_eta"]))
    df["sublead_sigmaE_over_E"] = df["sublead_energyErr"] / (
        df["sublead_pt"] * np.cosh(df["sublead_eta"])
    )

    df["deltaEta_g1_g2"] = df["lead_eta"] - df["sublead_eta"]
    df["deltaPhi_g1_g2"] = wrap_angle(df["lead_phi"] - df["sublead_phi"])

    df["deltaEta_gg_fj"] = df["eta"] - df["fatjet_selected_eta"]
    df["deltaPhi_gg_fj"] = wrap_angle(df["phi"] - df["fatjet_selected_phi"])
    df["deltaR_gg_fj"] = np.sqrt(df["deltaEta_gg_fj"] ** 2 + df["deltaPhi_gg_fj"] ** 2)

    df["deltaEta_g1_fj"] = df["lead_eta"] - df["fatjet_selected_eta"]
    df["deltaPhi_g1_fj"] = wrap_angle(df["lead_phi"] - df["fatjet_selected_phi"])
    df["deltaR_g1_fj"] = np.sqrt(df["deltaEta_g1_fj"] ** 2 + df["deltaPhi_g1_fj"] ** 2)

    df["deltaEta_g2_fj"] = df["sublead_eta"] - df["fatjet_selected_eta"]
    df["deltaPhi_g2_fj"] = wrap_angle(df["sublead_phi"] - df["fatjet_selected_phi"])
    df["deltaR_g2_fj"] = np.sqrt(df["deltaEta_g2_fj"] ** 2 + df["deltaPhi_g2_fj"] ** 2)

    df["deltaEta_subj1_gg"] = df["fatjet_selected_subjet1_eta"] - df["eta"]
    df["deltaPhi_subj1_gg"] = wrap_angle(df["fatjet_selected_subjet1_phi"] - df["phi"])
    df["deltaR_subj1_gg"] = np.sqrt(
        df["deltaEta_subj1_gg"] ** 2 + df["deltaPhi_subj1_gg"] ** 2
    )

    df["deltaEta_subj2_gg"] = df["fatjet_selected_subjet2_eta"] - df["eta"]
    df["deltaPhi_subj2_gg"] = wrap_angle(df["fatjet_selected_subjet2_phi"] - df["phi"])
    df["deltaR_subj2_gg"] = np.sqrt(
        df["deltaEta_subj2_gg"] ** 2 + df["deltaPhi_subj2_gg"] ** 2
    )

    df["deltaEta_subj1_subj2"] = (
        df["fatjet_selected_subjet1_eta"] - df["fatjet_selected_subjet2_eta"]
    )
    df["deltaPhi_subj1_subj2"] = wrap_angle(
        df["fatjet_selected_subjet1_phi"] - df["fatjet_selected_subjet2_phi"]
    )
    df["deltaR_subj1_subj2"] = np.sqrt(
        df["deltaEta_subj1_subj2"] ** 2 + df["deltaPhi_subj1_subj2"] ** 2
    )

    for photon in ["g1", "g2"]:
        eta_col = "lead_eta" if photon == "g1" else "sublead_eta"
        phi_col = "lead_phi" if photon == "g1" else "sublead_phi"
        for subjet in ["subj1", "subj2"]:
            subj_eta = f"fatjet_selected_{subjet.replace('subj', 'subjet')}_eta"
            subj_phi = f"fatjet_selected_{subjet.replace('subj', 'subjet')}_phi"
            df[f"deltaEta_{photon}_{subjet}"] = df[eta_col] - df[subj_eta]
            df[f"deltaPhi_{photon}_{subjet}"] = wrap_angle(df[phi_col] - df[subj_phi])
            df[f"deltaR_{photon}_{subjet}"] = np.sqrt(
                df[f"deltaEta_{photon}_{subjet}"] ** 2
                + df[f"deltaPhi_{photon}_{subjet}"] ** 2
            )

    df["DeltaR_jg_min"] = df[
        ["deltaR_g1_subj1", "deltaR_g1_subj2", "deltaR_g2_subj1", "deltaR_g2_subj2"]
    ].min(axis=1)

    vec_fj = vector.arr({
        "pt": df["fatjet_selected_pt"],
        "eta": df["fatjet_selected_eta"],
        "phi": df["fatjet_selected_phi"],
        "mass": df["fatjet_selected_msoftdrop"],
    })
    vec_gg = vector.arr({
        "pt": df["pt"],
        "eta": df["eta"],
        "phi": df["phi"],
        "mass": df["mass"],
    })
    df["mass_HH"] = (vec_fj + vec_gg).mass
    df["pt_HH"] = (vec_fj + vec_gg).pt
    df["eta_HH"] = (vec_fj + vec_gg).eta


def process_row_group(df, year):
    add_fatjet_base_variables(df)
    xbb_cols = choose_xbb_columns(df, year)

    pt_cols = [f"fatjet{i}_pt" for i in range(1, 5)]
    eta_cols = [f"fatjet{i}_eta" for i in range(1, 5)]
    mass_cols = [f"fatjet{i}_regmass" for i in range(1, 5)]
    tau_cols = [f"fatjet{i}_tau_ratio" for i in range(1, 5)]
    subjet1_eta_cols = [f"fatjet{i}_subjet1_eta" for i in range(1, 5)]
    subjet2_eta_cols = [f"fatjet{i}_subjet2_eta" for i in range(1, 5)]

    xbb_arr = df[xbb_cols].to_numpy(dtype=float)
    candidate_mask = (
        (df[pt_cols].to_numpy(dtype=float) > 300)
        & (df[tau_cols].to_numpy(dtype=float) < 0.75)
        & (np.abs(df[eta_cols].to_numpy(dtype=float)) <= 2.4)
        & (df[mass_cols].to_numpy(dtype=float) > 30)
        & (df[subjet1_eta_cols].to_numpy(dtype=float) > -99)
        & (df[subjet2_eta_cols].to_numpy(dtype=float) > -99)
    )
    event_mask = candidate_mask.any(axis=1)
    if not event_mask.any():
        return df.iloc[0:0].copy()

    df = df.loc[event_mask].copy()
    candidate_mask = candidate_mask[event_mask]
    xbb_arr = xbb_arr[event_mask]

    df["n_fatjets_final"] = candidate_mask.sum(axis=1)
    best_idx = np.argmax(np.where(candidate_mask, xbb_arr, -np.inf), axis=1)
    add_selected_fatjet(df, best_idx)
    add_btagging(df, year)
    add_bdt_variables(df)
    df["year"] = year
    return df


def process_file(input_path, output_dir):
    year = infer_year(input_path)
    parquet_file = pq.ParquetFile(input_path)
    pieces = []
    total_in = 0

    for row_group in range(parquet_file.num_row_groups):
        df = parquet_file.read_row_group(row_group).to_pandas()
        total_in += len(df)
        processed = process_row_group(df, year)
        if len(processed):
            pieces.append(processed)

    if pieces:
        out = pd.concat(pieces, ignore_index=True)
    else:
        empty_columns = list(parquet_file.schema_arrow.names)
        for col in DERIVED_COLUMNS_FOR_EMPTY_OUTPUT:
            if col not in empty_columns:
                empty_columns.append(col)
        out = pd.DataFrame(columns=empty_columns)

    rel_parent = input_path.parent.name
    output_subdir = output_dir / rel_parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / input_path.name.replace("_NOTAG_merged.parquet", "_Boost.parquet")
    out.to_parquet(output_path, index=False)
    print(f"{input_path.name}: {total_in} -> {len(out)}  saved {output_path}")
    return output_path, total_in, len(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply Boost_skim-style fatjet preselection to DataV6.")
    parser.add_argument("--input-dir", type=Path, default=DATA_V6_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=None, help="Debug option: process only first N files.")
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(args.input_dir.glob("Run*/*_NOTAG_merged.parquet"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No *_NOTAG_merged.parquet files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in files:
        summary.append(process_file(path, args.output_dir))

    summary_df = pd.DataFrame(summary, columns=["output_path", "n_input", "n_output"])
    summary_df["output_path"] = summary_df["output_path"].astype(str)
    summary_df.to_csv(args.output_dir / "processing_summary.csv", index=False)
    print(f"Processed {len(files)} files. Summary saved to {args.output_dir / 'processing_summary.csv'}")


if __name__ == "__main__":
    main()
