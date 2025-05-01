import numpy as np
import tifffile as tiff

def convert_tiff_to_raw(tiff_file_path, raw_file_path):
    # Read the TIFF file
    image = tiff.imread(tiff_file_path)
    
    # Ensure the image is 16-bit grayscale
    if image.dtype != np.uint16:
        raise ValueError("The input TIFF file is not a 16-bit grayscale image.")
    
    # Write the image data to a RAW file
    image.tofile(raw_file_path)
    print(f"Successfully converted {tiff_file_path} to {raw_file_path}.")

if __name__ == "__main__":
    # Example usage: replace the file paths with your actual file paths
    input_folder = "raw_input"
    output_folder = "raw_output"

    # Create output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert all TIFF files in the input folder to RAW format
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
            tiff_file_path = os.path.join(input_folder, filename)
            raw_file_path = os.path.join(output_folder, filename.replace('.tif', '.raw').replace('.tiff', '.raw'))
            
            try:
                convert_tiff_to_raw(tiff_file_path, raw_file_path)
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    print("Conversion completed!")