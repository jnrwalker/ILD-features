import SimpleITK as sitk
import os
import numpy as np
import glob
from fractal import fractal_dimension_3D
from tqdm import tqdm
import json
import argparse

def calculate_lobe_fractals(airway_directory, lobe_directory, output_file):
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
            
            # Get unique lobe labels 
            unique_lobes = np.unique(airway_lobe_np)
            unique_lobes = unique_lobes[unique_lobes != 0]
            
            # Calculate fractal dimension for each lobe
            for lobe_label in unique_lobes:
                mask_lobe = (airway_lobe_np == lobe_label)
                fd = float(fractal_dimension_3D(mask_lobe))
                patient_results["lobes"][f"lobe_{lobe_label}"] = fd
                print(f"Lobe {lobe_label} -> Fractal Dimension: {fd:.3f}")
            

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
    parser = argparse.ArgumentParser(description='Calculate fractal dimensions of airways by lung lobe')
    
    parser.add_argument('--airway_dir', type=str, default="outputs/segmentations",
                       help='Directory containing airway segmentation files')
    
    parser.add_argument('--lobe_dir', type=str, default="results/lobe_segmentations",
                       help='Directory containing lung lobe segmentation files')
    
    parser.add_argument('--output', type=str, default="results/per_lobe_fractal_dimensions.json",
                       help='Output JSON file to save results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Airway segmentation directory: {args.airway_dir}")
    print(f"Lobe segmentation directory: {args.lobe_dir}")
    print(f"Output file: {args.output}")
    
    calculate_lobe_fractals(args.airway_dir, args.lobe_dir, args.output)

if __name__ == "__main__":
    main() 



### To run the script:
### python3 per_lobe_fractal.py --airway_dir outputs/segmentations --lobe_dir results/lobe_segmentations --output results/per_lobe_fractal_dimensions.json
