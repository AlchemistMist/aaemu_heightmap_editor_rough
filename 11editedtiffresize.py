import numpy as np
import tifffile as tiff
import os
import sys

# --- Configuration ---
# !!! IMPORTANT: Set the desired output dimensions !!!
TARGET_WIDTH = 2048  # Replace with the desired width of the cropped image
TARGET_HEIGHT = 2048 # Replace with the desired height of the cropped image

INPUT_FOLDER = "edited_tiff"
OUTPUT_FOLDER = "edited_tiff_resized"
# --- End Configuration ---

def crop_tiff_bottom_left(input_path, output_path, target_width, target_height):
    """
    Reads a 16-bit grayscale TIFF, crops the bottom-left corner,
    and saves the cropped image.

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the cropped TIFF file.
        target_width (int): The desired width of the cropped image.
        target_height (int): The desired height of the cropped image.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Read the TIFF file using tifffile
        image = tiff.imread(input_path)

        # --- Validation ---
        # 1. Check data type
        if image.dtype != np.uint16:
            print(f"Warning: Skipping {os.path.basename(input_path)}. "
                  f"Image is not 16-bit (dtype is {image.dtype}).")
            return False

        # 2. Check dimensions (NumPy shape is height, width)
        original_height, original_width = image.shape
        if original_height < target_height or original_width < target_width:
            print(f"Warning: Skipping {os.path.basename(input_path)}. "
                  f"Image dimensions ({original_width}x{original_height}) are smaller than "
                  f"target crop size ({target_width}x{target_height}).")
            return False
        # --- End Validation ---

        # --- Cropping ---
        # Calculate slice indices for bottom-left corner
        # Rows: from (original_height - target_height) up to original_height
        # Columns: from 0 up to target_width
        # Using negative indexing for height is often cleaner: -target_height means start target_height rows from the end
        cropped_image = image[-target_height:, :target_width]
        # --- End Cropping ---

        # --- Saving ---
        # Save the cropped numpy array as a TIFF file
        tiff.imwrite(output_path, cropped_image)
        # --- End Saving ---

        print(f"Successfully cropped '{os.path.basename(input_path)}' ({original_width}x{original_height}) -> "
              f"'{os.path.basename(output_path)}' ({target_width}x{target_height})")
        return True

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while processing {os.path.basename(input_path)}: {e}")
        return False

if __name__ == "__main__":
    # Check if input folder exists
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        print("Please create the folder and place your .tif/.tiff files inside.")
        sys.exit(1) # Exit the script if the input folder is missing

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Starting TIFF cropping...")
    print(f"Input folder:  '{INPUT_FOLDER}'")
    print(f"Output folder: '{OUTPUT_FOLDER}'")
    print(f"Target crop size (WxH): {TARGET_WIDTH}x{TARGET_HEIGHT} pixels")
    print(f"Cropping region: Bottom-Left Corner")
    print(f"Expected format: 16-bit Grayscale TIFF")
    print("-" * 30)

    processed_count = 0
    skipped_count = 0
    error_count = 0
    file_count = 0

    # Process all TIFF files in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        # Check if the file ends with .tiff or .tif (case-insensitive)
        if filename.lower().endswith(('.tiff', '.tif')):
            file_count += 1
            input_file_path = os.path.join(INPUT_FOLDER, filename)
            # Output filename remains the same
            output_file_path = os.path.join(OUTPUT_FOLDER, filename)

            # Check if it's actually a file and not a directory
            if os.path.isfile(input_file_path):
                success = crop_tiff_bottom_left(input_file_path, output_file_path, TARGET_WIDTH, TARGET_HEIGHT)
                if success:
                    processed_count += 1
                else:
                    # Errors/skips are handled within the function's print statements
                    # Increment skip count if function returned False but didn't raise exception (usually warnings)
                    skipped_count +=1 # Count skips for clarity in summary
            else:
                 print(f"Skipping item (not a file): {filename}")
                 skipped_count += 1


    print("-" * 30)
    print("Cropping process finished.")
    print("\n--- Processing Summary ---")
    print(f"Total .tif/.tiff files found: {file_count}")
    print(f"Successfully processed and cropped: {processed_count}")
    print(f"Skipped (due to size/format warnings or not a file): {skipped_count}")
    # Note: error_count is implicitly covered by skipped_count here as the function returns False on errors too.
    print("-" * 30)