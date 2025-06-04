import os
import argparse
import json
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skan import Skeleton, summarize


def compute_tortuosity(df, length_threshold=1e-6):
    xyz_start = df[['coord-src-0', 'coord-src-1', 'coord-src-2']].to_numpy()
    xyz_end   = df[['coord-dst-0', 'coord-dst-1', 'coord-dst-2']].to_numpy()
    euclid = np.linalg.norm(xyz_start - xyz_end, axis=1)
    df = df.copy()
    df["euclid-length"] = euclid
    df_tort = df[df["euclid-length"] > length_threshold].copy()
    df_tort["tortuosity"] = df_tort["branch-distance"] / df_tort["euclid-length"]
    global_tort_weighted = (df_tort["branch-distance"] * df_tort["tortuosity"]).sum() / df_tort["branch-distance"].sum()
    global_tort_simple   = df_tort["tortuosity"].mean()
    return global_tort_simple, global_tort_weighted


def process_file(filepath):
    img = nib.load(filepath)
    seg = img.get_fdata().astype(bool)
    spacing = img.header.get_zooms()[:3]
    skeleton = skeletonize(seg)
    s = Skeleton(skeleton, spacing=spacing)
    df = summarize(s)
    # Ensure columns are named as expected (skan 0.13+ uses _ instead of -)
    if 'coord-src-0' not in df.columns:
        df = df.rename(columns={
            'coord_src_0': 'coord-src-0',
            'coord_src_1': 'coord-src-1',
            'coord_src_2': 'coord-src-2',
            'coord_dst_0': 'coord-dst-0',
            'coord_dst_1': 'coord-dst-1',
            'coord_dst_2': 'coord-dst-2',
            'branch_distance': 'branch-distance',
        })
    mean_tort, weighted_tort = compute_tortuosity(df)
    return mean_tort, weighted_tort


def main(input_dir, output_json):
    results = {}
    for fname in os.listdir(input_dir):
        if fname.endswith('.nii.gz'):
            identifier = os.path.splitext(os.path.splitext(fname)[0])[0]
            fpath = os.path.join(input_dir, fname)
            try:
                mean_tort, weighted_tort = process_file(fpath)
                results[identifier] = {
                    'mean_tortuosity': float(mean_tort),
                    'weighted_tortuosity': float(weighted_tort)
                }
                print(f"Processed {identifier}: mean={mean_tort:.4f}, weighted={weighted_tort:.4f}")
            except Exception as e:
                print(f"Error processing {identifier}: {e}")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch airway tree tortuosity analysis')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with NIfTI airway masks')
    parser.add_argument('--output_json', type=str, required=True, help='Output JSON file')
    args = parser.parse_args()
    main(args.input_dir, args.output_json) 