import os
import re
import numpy as np
import tifffile
from PIL import Image # Keep for resizing just in case
import math

# --- Configuration ---
INPUT_BASE_DIR = "output_sectors_cropped" # Folder containing XXX_YYY subfolders
OUTPUT_BASE_DIR = "sector_merged"  # Changed output dir name to reflect filtering
# Sector tiles are 32x32 after cropping
TARGET_SECTOR_WIDTH = 32
TARGET_SECTOR_HEIGHT = 32
# Keep resize filter in case some files were missed by cropping script or config changes
RESIZE_FILTER = Image.Resampling.BILINEAR

# --- Define the valid range for sector indices ---
MIN_INDEX = 0
MAX_INDEX = 15
# --- End Configuration ---

def merge_sectors_in_folder(input_folder, output_base_dir):
    """
    Finds all sector_YY_XX.tif files where YY and XX are between 0 and 15,
    resizes if needed, merges them into a single tif named after the
    input_folder's basename, saved in output_base_dir.
    """
    subfolder_name = os.path.basename(input_folder.rstrip(os.sep))
    print(f"Processing folder: '{subfolder_name}'")

    sector_files = {} # Stores {(x, y): filepath} for *valid* sectors
    max_valid_x = -1 # Max COLUMN index found *within the valid range*
    max_valid_y = -1 # Max ROW index found *within the valid range*

    sector_pattern = re.compile(r"sector_(\d+)_(\d+)\.tif$", re.IGNORECASE)

    # 1. Find VALID sector files and determine grid size based on them
    files_scanned = 0
    invalid_coords_found = 0
    try:
        for filename in os.listdir(input_folder):
            files_scanned += 1
            match = sector_pattern.match(filename)
            if match:
                yy_str = match.group(1)
                xx_str = match.group(2)
                try:
                    y = int(yy_str) # y coordinate = row index
                    x = int(xx_str) # x coordinate = column index
                except ValueError:
                     print(f"    Warning: Could not parse coordinates from '{filename}'. Skipping.")
                     continue # Skip this file

                # --- FILTERING LOGIC ---
                if MIN_INDEX <= x <= MAX_INDEX and MIN_INDEX <= y <= MAX_INDEX:
                    # This sector is within the desired 00-15 range
                    filepath = os.path.join(input_folder, filename)
                    sector_files[(x, y)] = filepath
                    max_valid_x = max(max_valid_x, x) # Update max VALID column found
                    max_valid_y = max(max_valid_y, y) # Update max VALID row found
                else:
                    # Coordinates are outside the 0-15 range
                    invalid_coords_found += 1
                    # print(f"    Skipping '{filename}': Coordinates ({x},{y}) outside range {MIN_INDEX}-{MAX_INDEX}.") # Optional verbose log
                # --- END FILTERING LOGIC ---

    except FileNotFoundError:
        print(f" Error: Input folder not found: {input_folder}")
        return
    except Exception as e:
        print(f" Error listing files in {input_folder}: {e}")
        return

    if invalid_coords_found > 0:
         print(f"  Skipped {invalid_coords_found} sector files with coordinates outside the {MIN_INDEX}-{MAX_INDEX} range.")

    if not sector_files:
        print(f"  No sector files within the range {MIN_INDEX}-{MAX_INDEX} found in '{subfolder_name}'. Skipping merge.")
        return

    # If we found valid files, max_valid_x/y should be >= 0
    if max_valid_x == -1 or max_valid_y == -1:
         # This case should ideally not happen if sector_files is not empty, but check anyway
         print(f"  Error: Found valid sector files but failed to determine grid dimensions for '{subfolder_name}'. Skipping.")
         return

    # 2. Calculate dimensions for the final merged image BASED ON VALID MAX INDICES
    # The grid will span from 0,0 up to max_valid_x, max_valid_y
    grid_width_tiles = max_valid_x + 1
    grid_height_tiles = max_valid_y + 1
    total_width = grid_width_tiles * TARGET_SECTOR_WIDTH
    total_height = grid_height_tiles * TARGET_SECTOR_HEIGHT
    print(f"  Valid Grid: {grid_width_tiles} columns x {grid_height_tiles} rows ({len(sector_files)} sectors to merge).")
    print(f"  Output image size: {total_width}x{total_height} pixels.")

    # 3. Create the blank canvas
    merged_image_canvas = np.zeros((total_height, total_width), dtype=np.uint16)

    # 4. Iterate through VALID sectors, load, resize (if needed), and place onto canvas
    processed_count = 0
    skipped_resize_count = 0
    for (x, y), filepath in sector_files.items(): # This loop only includes files within 0-15 range
        try:
            sector_data = tifffile.imread(filepath)

            if sector_data.dtype != np.uint16:
                 print(f"    Warning: Sector col={x}, row={y} ({filepath}) is not uint16 ({sector_data.dtype}). Converting.")
                 sector_data = sector_data.astype(np.uint16)

            current_height, current_width = sector_data.shape

            # Check if resizing is needed
            if current_width != TARGET_SECTOR_WIDTH or current_height != TARGET_SECTOR_HEIGHT:
                print(f"    Resizing sector col={x}, row={y} from {current_width}x{current_height} to {TARGET_SECTOR_WIDTH}x{TARGET_SECTOR_HEIGHT}...")
                pil_img = Image.fromarray(sector_data)
                resized_pil_img = pil_img.resize((TARGET_SECTOR_WIDTH, TARGET_SECTOR_HEIGHT), RESIZE_FILTER)
                final_sector_data = np.array(resized_pil_img).astype(np.uint16)
                skipped_resize_count +=1
            else:
                final_sector_data = sector_data

            # Calculate placement offsets (still based on 0-15 indices)
            x_offset = x * TARGET_SECTOR_WIDTH
            y_offset = y * TARGET_SECTOR_HEIGHT

            # Place the sector data onto the canvas
            merged_image_canvas[y_offset:y_offset + TARGET_SECTOR_HEIGHT,
                                x_offset:x_offset + TARGET_SECTOR_WIDTH] = final_sector_data
            processed_count += 1

        except FileNotFoundError:
             print(f"    Error: File not found during processing: {filepath}")
             continue
        except Exception as e:
            print(f"    Error processing file {filepath} (col={x}, row={y}): {e}")
            continue
    if skipped_resize_count > 0:
        print(f"    Note: Had to resize {skipped_resize_count} sectors to fit target dimensions.")


    # 5. Save the merged image (using only valid sectors)
    if processed_count > 0:
        try:
            output_filename = f"{subfolder_name}.tif"
            output_filepath = os.path.join(output_base_dir, output_filename)
            print(f"  Saving merged map ({processed_count} sectors from 0-15 range) to: {output_filepath}")
            tifffile.imwrite(output_filepath, merged_image_canvas)
        except Exception as e:
            print(f"  Error saving merged file {output_filepath}: {e}")
    else:
        # This case should be caught earlier, but good to have
        print(f"  No valid sectors ({MIN_INDEX}-{MAX_INDEX}) were successfully processed for '{subfolder_name}'. Merged file not saved.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting sector merge process (Filtering for 00-15 sectors)...")
    print(f"Input base directory: {INPUT_BASE_DIR}")
    print(f"Output base directory: {OUTPUT_BASE_DIR}")
    print(f"Including sectors with indices from {MIN_INDEX} to {MAX_INDEX} (inclusive).")
    print(f"Sector tile size expected: {TARGET_SECTOR_WIDTH}x{TARGET_SECTOR_HEIGHT}")

    if not os.path.isdir(INPUT_BASE_DIR):
        print(f"Error: Input directory '{INPUT_BASE_DIR}' not found.")
        exit()

    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    except OSError as e:
         print(f"Error creating output base directory '{OUTPUT_BASE_DIR}': {e}")
         exit()

    subfolder_count = 0
    processed_subfolder_count = 0
    for item in os.listdir(INPUT_BASE_DIR):
        input_subfolder_path = os.path.join(INPUT_BASE_DIR, item)
        if os.path.isdir(input_subfolder_path):
            subfolder_count += 1
            merge_sectors_in_folder(input_subfolder_path, OUTPUT_BASE_DIR)
            processed_subfolder_count +=1 # Assuming func handles its own errors for skipping

    if subfolder_count == 0:
         print(f"\nNo subdirectories found to process within '{INPUT_BASE_DIR}'.")
    else:
        print(f"\nFinished iterating through {processed_subfolder_count} subfolders.")

    print("\nSector merge process finished.")