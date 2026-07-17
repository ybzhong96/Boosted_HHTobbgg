import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vector


features = [
    "lead_eta",
    "eta",
    "lead_mvaID",
    "sublead_mvaID",
    'Res_DNNpair_CosThetaStar_gg',
    'Res_DNNpair_CosThetaStar_CS',
    'Res_DNNpair_pholead_PtOverM',
    'Res_DNNpair_phosublead_PtOverM',
    'lead_sigmaE_over_E',
    'sublead_sigmaE_over_E',

    'n_leptons',
    'lepton1_pfIsoId',
    'lepton1_pt',
    'puppiMET_pt',
    'puppiMET_sumEt',

    'fatjet_selected_regmass',
    'fatjet_selected_tau21',
    'fatjet_selected_tau32',
    'b-tagging',
    'fatjet_selected_pt',

    'DeltaR_jg_min',
    'deltaR_subj1_subj2',
    'deltaEta_gg_fj',
    'deltaPhi_gg_fj',
    'deltaR_gg_fj',
    'n_jets',
    'n_fatjets',
    'n_fatjets_final'
]


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def write_features_list(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "features_list.txt").write_text("\n".join(features) + "\n")


def process_file(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Processing {file_path}")
    df = pd.read_pickle(file_path)

    df['lead_sigmaE_over_E'] = df['lead_energyErr'] / (df['lead_pt'] * np.cosh(df['lead_eta']))
    df['sublead_sigmaE_over_E'] = df['sublead_energyErr'] / (
        df['sublead_pt'] * np.cosh(df['sublead_eta'])
    )

    print("Define shorthand references")
    g1_eta = df['lead_eta']
    g1_phi = df['lead_phi']
    g2_eta = df['sublead_eta']
    g2_phi = df['sublead_phi']
    gg_eta = df['eta']  # entire diphoton system
    gg_phi = df['phi']  # entire diphoton system
    fj_eta = df['fatjet_selected_eta']
    fj_phi = df['fatjet_selected_phi']
    subj1_eta = df['fatjet_selected_subjet1_eta']
    subj1_phi = df['fatjet_selected_subjet1_phi']
    subj2_eta = df['fatjet_selected_subjet2_eta']
    subj2_phi = df['fatjet_selected_subjet2_phi']

    df['deltaEta_g1_g2'] = g1_eta - g2_eta
    df['deltaPhi_g1_g2'] = wrap_angle(g1_phi - g2_phi)

    df['deltaEta_gg_fj'] = gg_eta - fj_eta
    df['deltaPhi_gg_fj'] = wrap_angle(gg_phi - fj_phi)
    df['deltaR_gg_fj'] = np.sqrt(df['deltaEta_gg_fj']**2 + df['deltaPhi_gg_fj']**2)

    df['deltaEta_subj1_subj2'] = subj1_eta - subj2_eta
    df['deltaPhi_subj1_subj2'] = wrap_angle(subj1_phi - subj2_phi)
    df['deltaR_subj1_subj2'] = np.sqrt(
        df['deltaEta_subj1_subj2']**2 + df['deltaPhi_subj1_subj2']**2
    )

    df['deltaEta_g1_subj1'] = g1_eta - subj1_eta
    df['deltaPhi_g1_subj1'] = wrap_angle(g1_phi - subj1_phi)
    df['deltaR_g1_subj1'] = np.sqrt(df['deltaEta_g1_subj1']**2 + df['deltaPhi_g1_subj1']**2)

    df['deltaEta_g1_subj2'] = g1_eta - subj2_eta
    df['deltaPhi_g1_subj2'] = wrap_angle(g1_phi - subj2_phi)
    df['deltaR_g1_subj2'] = np.sqrt(df['deltaEta_g1_subj2']**2 + df['deltaPhi_g1_subj2']**2)

    df['deltaEta_g2_subj1'] = g2_eta - subj1_eta
    df['deltaPhi_g2_subj1'] = wrap_angle(g2_phi - subj1_phi)
    df['deltaR_g2_subj1'] = np.sqrt(df['deltaEta_g2_subj1']**2 + df['deltaPhi_g2_subj1']**2)

    df['deltaEta_g2_subj2'] = g2_eta - subj2_eta
    df['deltaPhi_g2_subj2'] = wrap_angle(g2_phi - subj2_phi)
    df['deltaR_g2_subj2'] = np.sqrt(df['deltaEta_g2_subj2']**2 + df['deltaPhi_g2_subj2']**2)

    df['DeltaR_jg_min'] = df[
        ["deltaR_g1_subj1", "deltaR_g1_subj2", "deltaR_g2_subj1", "deltaR_g2_subj2"]
    ].min(axis=1)

    if 'fatjet_selected_globalParT3_XbbVsQCD' in df.columns:
        xbb = df['fatjet_selected_globalParT3_XbbVsQCD']
        xbb_bins = [
            ((xbb >= 0.57) & (xbb < 0.83), 1),
            ((xbb >= 0.8) & (xbb < 0.86), 2),
            ((xbb >= 0.86) & (xbb < 0.91), 3),
            ((xbb >= 0.91) & (xbb < 0.96), 4),
            (xbb >= 0.96, 5),
        ]
    else:
        xbb = df['fatjet_selected_particleNet_XbbVsQCD']
        xbb_bins = [
            ((xbb >= 0.4) & (xbb < 0.83), 1),
            ((xbb >= 0.83) & (xbb < 0.89), 2),
            ((xbb >= 0.89) & (xbb < 0.925), 3),
            ((xbb >= 0.925) & (xbb < 0.96), 4),
            (xbb >= 0.96, 5),
        ]

    df['b-tagging'] = 0
    for mask, value in xbb_bins:
        df.loc[mask, 'b-tagging'] = value

    vec1 = vector.arr({
        "pt": df["fatjet_selected_pt"],
        "eta": df["fatjet_selected_eta"],
        "phi": df["fatjet_selected_phi"],
        "mass": df["fatjet_selected_regmass"]
    })
    vec2 = vector.arr({
        "pt": df["pt"],
        "eta": df["eta"],
        "phi": df["phi"],
        "mass": df["mass"]
    })
    df.loc[:, "mass_HH"] = (vec1 + vec2).mass
    df.loc[:, "pt_HH"] = (vec1 + vec2).pt
    df.loc[:, "eta_HH"] = (vec1 + vec2).eta

    output_dir = file_path.parent / "output_parquet"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{file_path.stem}.parquet"

    print(output_file)
    df.to_parquet(output_file, index=False)
    write_features_list(output_dir)


def get_input_files(args):
    if len(args) == 0:
        return sorted(Path(".").glob("Boost_*.pkl"))

    input_files = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            input_files.extend(sorted(path.glob("Boost_*.pkl")))
        else:
            input_files.append(path)
    return input_files


if __name__ == "__main__":
    files = get_input_files(sys.argv[1:])
    if not files:
        print("No Boost_*.pkl files found.")
        sys.exit(1)

    print(f"Found {len(files)} file(s).")
    for file_path in files:
        process_file(file_path)
