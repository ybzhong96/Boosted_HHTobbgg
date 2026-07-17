import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import vector


FEATURES = [
"fatjet_selected_regmass",
"fatjet_selected_tau21",
"fatjet_selected_Xbb_wp2",
"fatjet_selected_Xbb_wp3",
"fatjet_selected_Xbb_wp4",
"fatjet_selected_Xbb_wp5",
"fatjet_selected_pt",
"lead_eta",
"eta",
"lead_mvaID",
"sublead_mvaID",
"n_leptons",
"Res_DNNpair_CosThetaStar_gg",
"Res_DNNpair_pholead_PtOverM",
"Res_DNNpair_phosublead_PtOverM",
"DeltaR_jg_min",
"deltaEta_g1_g2",
"deltaEta_gg_fj",
"deltaR_gg_fj",
"deltaEta_g1_fj",
"deltaR_g1_fj",
"deltaEta_g2_fj",
"deltaR_g2_fj",
"deltaEta_subj1_gg",
"deltaR_subj1_gg",
"deltaEta_subj2_gg",
"deltaR_subj2_gg",
"deltaR_subj1_subj2",
"n_jets",
"n_fatjets",
"n_fatjets_final"
]

EXTRA_OUTPUT_FEATURES = [
    "b-tagging",
    "fatjet_selected_tau21",
    "fatjet_selected_tau32",
    "fatjet_selected_globalParT3_XbbVsQCD",
    "mass_HH",
    "pt_HH",
    "eta_HH",
    "lead_sigmaE_over_E",
    "sublead_sigmaE_over_E",
]


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def write_features_list(output_dir, features=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = list(FEATURES if features is None else features)

    txt_path = output_dir / "features_list.txt"
    json_path = output_dir / "features_list.json"
    txt_path.write_text("\n".join(features) + "\n")
    json_path.write_text(json.dumps(features, indent=2) + "\n")
    return txt_path, json_path


def add_reference_features(df):
    g1_eta = df["lead_eta"]
    g1_phi = df["lead_phi"]
    g2_eta = df["sublead_eta"]
    g2_phi = df["sublead_phi"]
    gg_eta = df["eta"]
    gg_phi = df["phi"]
    fj_eta = df["fatjet_selected_eta"]
    fj_phi = df["fatjet_selected_phi"]
    subj1_eta = df["fatjet_selected_subjet1_eta"]
    subj1_phi = df["fatjet_selected_subjet1_phi"]
    subj2_eta = df["fatjet_selected_subjet2_eta"]
    subj2_phi = df["fatjet_selected_subjet2_phi"]

    df["deltaEta_g1_g2"] = g1_eta - g2_eta
    df["deltaPhi_g1_g2"] = wrap_angle(g1_phi - g2_phi)

    df["deltaEta_gg_fj"] = gg_eta - fj_eta
    df["deltaPhi_gg_fj"] = wrap_angle(gg_phi - fj_phi)
    df["deltaR_gg_fj"] = np.sqrt(df["deltaEta_gg_fj"] ** 2 + df["deltaPhi_gg_fj"] ** 2)

    df["deltaEta_g1_fj"] = g1_eta - fj_eta
    df["deltaPhi_g1_fj"] = wrap_angle(g1_phi - fj_phi)
    df["deltaR_g1_fj"] = np.sqrt(df["deltaEta_g1_fj"] ** 2 + df["deltaPhi_g1_fj"] ** 2)

    df["deltaEta_g2_fj"] = g2_eta - fj_eta
    df["deltaPhi_g2_fj"] = wrap_angle(g2_phi - fj_phi)
    df["deltaR_g2_fj"] = np.sqrt(df["deltaEta_g2_fj"] ** 2 + df["deltaPhi_g2_fj"] ** 2)

    df["deltaEta_subj1_gg"] = subj1_eta - gg_eta
    df["deltaPhi_subj1_gg"] = wrap_angle(subj1_phi - gg_phi)
    df["deltaR_subj1_gg"] = np.sqrt(df["deltaEta_subj1_gg"] ** 2 + df["deltaPhi_subj1_gg"] ** 2)

    df["deltaEta_subj2_gg"] = subj2_eta - gg_eta
    df["deltaPhi_subj2_gg"] = wrap_angle(subj2_phi - gg_phi)
    df["deltaR_subj2_gg"] = np.sqrt(df["deltaEta_subj2_gg"] ** 2 + df["deltaPhi_subj2_gg"] ** 2)

    df["deltaEta_subj1_subj2"] = subj1_eta - subj2_eta
    df["deltaPhi_subj1_subj2"] = wrap_angle(subj1_phi - subj2_phi)
    df["deltaR_subj1_subj2"] = np.sqrt(
        df["deltaEta_subj1_subj2"] ** 2 + df["deltaPhi_subj1_subj2"] ** 2
    )

    df["deltaEta_g1_subj1"] = g1_eta - subj1_eta
    df["deltaPhi_g1_subj1"] = wrap_angle(g1_phi - subj1_phi)
    df["deltaR_g1_subj1"] = np.sqrt(df["deltaEta_g1_subj1"] ** 2 + df["deltaPhi_g1_subj1"] ** 2)

    df["deltaEta_g1_subj2"] = g1_eta - subj2_eta
    df["deltaPhi_g1_subj2"] = wrap_angle(g1_phi - subj2_phi)
    df["deltaR_g1_subj2"] = np.sqrt(df["deltaEta_g1_subj2"] ** 2 + df["deltaPhi_g1_subj2"] ** 2)

    df["deltaEta_g2_subj1"] = g2_eta - subj1_eta
    df["deltaPhi_g2_subj1"] = wrap_angle(g2_phi - subj1_phi)
    df["deltaR_g2_subj1"] = np.sqrt(df["deltaEta_g2_subj1"] ** 2 + df["deltaPhi_g2_subj1"] ** 2)

    df["deltaEta_g2_subj2"] = g2_eta - subj2_eta
    df["deltaPhi_g2_subj2"] = wrap_angle(g2_phi - subj2_phi)
    df["deltaR_g2_subj2"] = np.sqrt(df["deltaEta_g2_subj2"] ** 2 + df["deltaPhi_g2_subj2"] ** 2)

    df["DeltaR_jg_min"] = df[
        ["deltaR_g1_subj1", "deltaR_g1_subj2", "deltaR_g2_subj1", "deltaR_g2_subj2"]
    ].min(axis=1)

    if "fatjet_selected_tau21" in df.columns:
        df["fatjet_selected_tau2tau1_ratio"] = df["fatjet_selected_tau21"]
    elif "fatjet_selected_tau2" in df.columns and "fatjet_selected_tau1" in df.columns:
        df["fatjet_selected_tau2tau1_ratio"] = df["fatjet_selected_tau2"] / df["fatjet_selected_tau1"]

    if "fatjet_selected_globalParT3_XbbVsQCD" in df.columns:
        xbb = df["fatjet_selected_globalParT3_XbbVsQCD"]
    else:
        xbb = df["fatjet_selected_particleNet_XbbVsQCD"]
    df["fatjet_selected_Xbb_wp2"] = ((xbb >= 0.5) & (xbb < 0.8)).astype(int)
    df["fatjet_selected_Xbb_wp3"] = ((xbb >= 0.8) & (xbb < 0.9)).astype(int)
    df["fatjet_selected_Xbb_wp4"] = ((xbb >= 0.9) & (xbb < 0.95)).astype(int)
    df["fatjet_selected_Xbb_wp5"] = (xbb >= 0.95).astype(int)

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
    return df


def read_input(path):
    path = Path(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {path}")


def process_file(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(input_path)
    df = add_reference_features(df)
    output_path = output_dir / f"{input_path.stem}_ref.parquet"
    df.to_parquet(output_path, index=False)
    write_features_list(output_dir)
    print(output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Add reference BDT features to pkl/parquet files.")
    parser.add_argument("inputs", nargs="+", help="Input pkl/parquet file(s).")
    parser.add_argument("--output-dir", default="output_parquet_ref")
    return parser.parse_args()


def main():
    args = parse_args()
    for input_path in args.inputs:
        process_file(input_path, args.output_dir)


if __name__ == "__main__":
    main()
