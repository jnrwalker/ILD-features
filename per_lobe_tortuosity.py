#!/usr/bin/env python3
import SimpleITK as sitk
import os
import numpy as np
import glob
from tqdm import tqdm
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import json
import argparse

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

def calculate_lobe_tortuosity(airway_directory, lobe_directory, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all airway segmentation files with RBH prefix
    airway_files = glob.glob(os.path.join(airway_directory, "RBH*.nii.gz"))
    print(f"Found {len(airway_files)} airway segmentation files")
    
    all_results = []

    for airway_file in tqdm(airway_files, desc="Processing cases"):
        patient_id = os.path.basename(airway_file).split('.')[0]
        lobe_file = os.path.join(lobe_directory, patient_id, f"{patient_id}_lobes.nii.gz")
        
        if not os.path.exists(lobe_file):
            print(f"WARNING: Lobe segmentation not found for {patient_id}, skipping")
            continue
        
        print(f"\nProcessing {patient_id}")
        
        # Load the airway and lobe segmentation images
        try:
            airway_image = sitk.ReadImage(airway_file, sitk.sitkUInt8)
            lobe_image = sitk.ReadImage(lobe_file, sitk.sitkUInt8)
            
            
            airway_array = sitk.GetArrayFromImage(airway_image)
            lobe_array = sitk.GetArrayFromImage(lobe_image)
            
      
            airway_lobe_np = lobe_array * airway_array
            

            patient_results = {
                "patient_id": patient_id,
                "lobes": {}
            }
            
            # Get unique lobe labels (excluding background)
            unique_lobes = np.unique(airway_lobe_np)
            unique_lobes = unique_lobes[unique_lobes != 0]
            
            # Calculate tortuosity for each lobe
            for lobe_label in unique_lobes:
       
                mask_lobe = (airway_lobe_np == lobe_label)
                
       
                if not np.any(mask_lobe):
                    continue
                
 
                spacing = airway_image.GetSpacing()
                skeleton = skeletonize(mask_lobe)
                s = Skeleton(skeleton, spacing=spacing)
                df = summarize(s)
                
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
                
                # Calculate tortuosity
                mean_tort, weighted_tort = compute_tortuosity(df)
                
                patient_results["lobes"][f"lobe_{lobe_label}"] = {
                    "mean_tortuosity": float(mean_tort),
                    "weighted_tortuosity": float(weighted_tort)
                }
                print(f"Lobe {lobe_label} -> Mean Tortuosity: {mean_tort:.3f}, Weighted Tortuosity: {weighted_tort:.3f}")
            
            all_results.append(patient_results)
            
        except Exception as e:
            print(f"Error processing {patient_id}: {str(e)}")
            continue
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nProcessed {len(all_results)} cases successfully")
    print(f"Results saved to {output_file}")
    
    return all_results

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate tortuosity of airways by lung lobe')
    
    parser.add_argument('--airway_dir', type=str, default="outputs/segmentations",
                       help='Directory containing airway segmentation files')
    
    parser.add_argument('--lobe_dir', type=str, default="results/lobe_segmentations",
                       help='Directory containing lung lobe segmentation files')
    
    parser.add_argument('--output', type=str, default="results/per_lobe_tortuosity.json",
                       help='Output JSON file to save results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Airway segmentation directory: {args.airway_dir}")
    print(f"Lobe segmentation directory: {args.lobe_dir}")
    print(f"Output file: {args.output}")
    
    calculate_lobe_tortuosity(args.airway_dir, args.lobe_dir, args.output)

if __name__ == "__main__":
    main()

### To run the script:
### python3 per_lobe_tortuosity.py --airway_dir airways/segmentations --lobe_dir results/lobe_segmentations --output results/per_lobe_tortuosity.json
