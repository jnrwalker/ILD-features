import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import argparse
import json
from pathlib import Path
from tqdm import tqdm

def fractal_dimension_3D(array, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0, plot=False):

    if max_box_size == None:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
        
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
    scales = np.unique(scales)  # remove duplicates that could occur as a result of the floor

    # get the locations of all non-zero pixels
    locs = np.where(array > 0)
    voxels = np.array([(x, y, z) for x, y, z in zip(*locs)])

    # count the minimum amount of boxes touched
    Ns = []
    
    # loop over all scales
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        # search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(0, i, scale) for i in array.shape]
            bin_edges = [np.hstack([0 - offset, x + offset]) for x in bin_edges]
            H1, e = np.histogramdd(voxels, bins=bin_edges)
            touched.append(np.sum(H1 > 0))
        Ns.append(touched)
    Ns = np.array(Ns)

    Ns = Ns.min(axis=1)

    scales = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])

    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[:len(Ns)]
    
    # perform fit
    coeffs = np.polyfit(np.log(1/scales), np.log(Ns), 1)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(np.log(1/scales), np.log(Ns), 'bo-', label='Data points')
        plt.plot(np.log(1/scales), coeffs[1] + coeffs[0]*np.log(1/scales), 'r-', 
                label=f'Fit (slope = {coeffs[0]:.3f})')
        plt.xlabel('log(1/scale)')
        plt.ylabel('log(N)')
        plt.title('Log-log plot of box counting fractal dimension')
        plt.legend()
        plt.grid(True)
        plt.show()

    return coeffs[0]

def process_folder(input_dir, output_json):
    """Process all .nii.gz files in a folder and calculate their fractal dimensions"""
    input_path = Path(input_dir)
    results = {}
    
    # Get all .nii.gz files
    nifti_files = list(input_path.glob('*.nii.gz'))
    
    # Process each file with a progress bar
    for nifti_file in tqdm(nifti_files, desc="Processing files"):
        try:
            # Read the image
            image = sitk.ReadImage(str(nifti_file))
            array = sitk.GetArrayFromImage(image)
            
            # Calculate fractal dimension
            fd = fractal_dimension_3D(array, plot=False)
            
            # Store result using filename without extension as key
            case_number = nifti_file.stem.replace('.nii', '')
            results[case_number] = float(fd)  # Convert to float for JSON serialization
            
        except Exception as e:
            print(f"Error processing {nifti_file.name}: {str(e)}")
    
    # Save results to JSON file
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_json}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate fractal dimensions for medical image segmentations')
    parser.add_argument('input_dir', type=str, help='Directory containing .nii.gz files')
    parser.add_argument('--output', type=str, default='fractal_dimensions.json',
                        help='Output JSON file path (default: fractal_dimensions.json)')
    
    args = parser.parse_args()
    
    # Process the folder
    results = process_folder(args.input_dir, args.output)
    
    # Print summary
    print(f"\nProcessed {len(results)} files")

if __name__ == "__main__":
    main()


####### How to run:
#  python3 fractal.py outputs/segementations --output fractal_dim.json
