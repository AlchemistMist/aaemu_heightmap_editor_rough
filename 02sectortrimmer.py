import os
import sys
from PIL import Image, UnidentifiedImageError

# --- Configuration ---
INPUT_BASE_DIR = "output_sectors"  # Folder containing subfolders with images
OUTPUT_BASE_DIR = "output_sectors_cropped" # Main folder for all cropped sector outputs

# Define the mapping of allowed input sizes (W, H) to output sizes (W-1, H-1)
# The script will ONLY process images whose dimensions match a key in this map.
CROP_MAP = {
    (33, 33): (32, 32),
    (17, 17): (16, 16),
    (9, 9):   (8, 8),
    (5, 5):   (4, 4),
    (3, 3):   (2, 2), 
    (2, 2):   (1, 1),
}
# --- End Configuration ---

def crop_single_image(input_path, output_path):
    """
    Opens an image, checks if its dimensions are in CROP_MAP,
    crops it by 1 pixel (W-1, H-1), and saves it.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path where the cropped image will be saved.

    Returns:
        int: 1 if successfully cropped and saved.
             0 if skipped due to dimensions not being in CROP_MAP.
            -1 if a critical error occurred (e.g., file not found, cannot open).
    """
    try:
        with Image.open(input_path) as img:
            img_w, img_h = img.size
            input_dims = (img_w, img_h)

            # --- Validation against CROP_MAP ---
            if input_dims in CROP_MAP:
                # Get the target dimensions from the map
                target_w, target_h = CROP_MAP[input_dims]
                # print(f"  -> Cropping '{os.path.basename(input_path)}' from {img_w}x{img_h} to {target_w}x{target_h}") # Optional verbose log
            else:
                # Dimensions do not match any allowed input size
                print(f"  -> Skipping '{os.path.basename(input_path)}': Size {img_w}x{img_h} not in allowed crop list {list(CROP_MAP.keys())}.")
                return 0 # Indicate skipped

            # --- Cropping ---
            # Crop box (left, upper, right, lower)
            # Keeps pixels from 0 up to (but not including) target_w/target_h
            crop_box = (0, 0, target_w, target_h)
            cropped_img = img.crop(crop_box)

            # --- Saving ---
            # Save the cropped tile as uncompressed TIFF, preserving mode if possible
            cropped_img.save(output_path, format='TIFF', compression='None')
            return 1 # Indicate success

    except FileNotFoundError:
        print(f"  -> Error: Input file not found: '{input_path}'. Skipping.")
        return -1 # Indicate critical error
    except UnidentifiedImageError:
        print(f"  -> Error: Cannot identify '{os.path.basename(input_path)}' as an image file. Skipping.")
        return -1 # Indicate critical error
    except Exception as e:
        print(f"  -> Error processing or saving image '{os.path.basename(input_path)}': {e}")
        return -1 # Indicate critical error


def process_folders(input_base_dir, output_base_dir):
    """
    Iterates through subfolders in input_base_dir, finds TIFF images,
    crops them according to CROP_MAP, and saves them to corresponding
    subfolders in output_base_dir.
    """
    # --- 1. Validate Input and Create Main Output Directory ---
    if not os.path.isdir(input_base_dir):
        print(f"Error: Input base directory '{input_base_dir}' not found.")
        sys.exit(1)

    try:
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"Input base directory: '{input_base_dir}'")
        print(f"Output base directory: '{output_base_dir}'")
        print(f"Processing images with input sizes: {list(CROP_MAP.keys())}")
        print("Cropping each matching image by 1 pixel in width and height.")
    except OSError as e:
        print(f"Error creating output base directory '{output_base_dir}': {e}")
        sys.exit(1)

    # --- 2. Process Subfolders and Files ---
    total_processed_count = 0
    total_skipped_count = 0
    total_error_count = 0

    print("\nStarting image cropping...")

    try:
        subfolders = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    except OSError as e:
        print(f"Error accessing input base directory '{input_base_dir}': {e}")
        sys.exit(1)

    if not subfolders:
        print("No subdirectories found in the input directory.")
        return

    for subfolder_name in subfolders:
        input_subfolder_path = os.path.join(input_base_dir, subfolder_name)
        output_subfolder_path = os.path.join(output_base_dir, subfolder_name)

        print(f"\nProcessing subfolder: '{subfolder_name}'")

        try:
            os.makedirs(output_subfolder_path, exist_ok=True)
        except OSError as e:
            print(f"  -> Error creating output sub-directory '{output_subfolder_path}'. Skipping this subfolder: {e}")
            continue

        try:
             files_in_subfolder = os.listdir(input_subfolder_path)
        except OSError as e:
             print(f"  -> Error listing files in '{input_subfolder_path}'. Skipping this subfolder: {e}")
             continue

        sub_processed = 0
        sub_skipped = 0
        sub_errors = 0

        for filename in files_in_subfolder:
            if os.path.isfile(os.path.join(input_subfolder_path, filename)) and filename.lower().endswith(('.tif', '.tiff')):
                input_image_path = os.path.join(input_subfolder_path, filename)
                output_image_path = os.path.join(output_subfolder_path, filename)

                result = crop_single_image(input_image_path, output_image_path)

                if result == 1:
                    sub_processed += 1
                elif result == 0:
                    sub_skipped += 1
                elif result == -1:
                    sub_errors += 1
            # else: # Optional: count non-TIFF files as skipped
            #    if os.path.isfile(os.path.join(input_subfolder_path, filename)):
            #        sub_skipped += 1

        print(f"  Subfolder summary: Processed={sub_processed}, Skipped={sub_skipped}, Errors={sub_errors}")
        total_processed_count += sub_processed
        total_skipped_count += sub_skipped
        total_error_count += sub_errors

    # --- 3. Print Overall Summary ---
    print("\n--- Cropping Summary ---")
    print(f"Total images successfully cropped and saved: {total_processed_count}")
    print(f"Total images skipped (size not in allowed list): {total_skipped_count}")
    print(f"Total images failed due to processing errors: {total_error_count}")
    print("------------------------")


# --- Main execution ---
if __name__ == "__main__":
    print("--- Multi-Size Image Cropping Script ---")
    process_folders(INPUT_BASE_DIR, OUTPUT_BASE_DIR)
    print("\nScript finished.")