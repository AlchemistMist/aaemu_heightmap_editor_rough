import numpy as np
import tifffile as tiff
import os
import sys

# --- Configuration ---
INPUT_FOLDER = "edited_cells"       # Folder containing XXX_YYY.tif or fXXX_YYY.tif files
OUTPUT_FOLDER = "edited_sectors"    # Main output folder for sector subdirectories

# Sector dimensions
BASE_SECTOR_WIDTH = 32
BASE_SECTOR_HEIGHT = 32
SECTOR_OVERLAP = 1
# Target output size (including overlap), may be smaller for edge 'f' cases
OUTPUT_SECTOR_WIDTH = BASE_SECTOR_WIDTH + SECTOR_OVERLAP  # 33
OUTPUT_SECTOR_HEIGHT = BASE_SECTOR_HEIGHT + SECTOR_OVERLAP # 33

# Expected number of sectors per cell (assuming a grid based on BASE dimensions)
NUM_SECTORS_WIDE = 16
NUM_SECTORS_HIGH = 16

# --- End Configuration ---

def process_cell_image(input_cell_path, output_base_folder):
    """
    Reads a cell image, determines its type (513x513 standard, or 512x512,
    512x513, 513x512 'f' type edge cells), and splits it into 16x16 sectors
    with overlap, handling edge cases and applying 'f' prefix to output sectors
    on edges missing overlap.

    Args:
        input_cell_path (str): Full path to the input cell TIFF file.
        output_base_folder (str): Path to the main 'edited_sectors' directory.

    Returns:
        tuple: (success_count, failure_count) for sectors from this cell image.
    """
    cell_filename = os.path.basename(input_cell_path)
    cell_base_name = os.path.splitext(cell_filename)[0] # Filename without .tif

    print(f"\nProcessing Cell: {cell_filename}")
    success_count = 0
    failure_count = 0
    expected_total_sectors = NUM_SECTORS_WIDE * NUM_SECTORS_HIGH

    # --- Create Output Subfolder ---
    output_subfolder = os.path.join(output_base_folder, cell_base_name)
    try:
        os.makedirs(output_subfolder, exist_ok=True)
    except OSError as e:
        print(f"  Error: Could not create output subfolder '{output_subfolder}': {e}")
        return 0, expected_total_sectors # Count all potential sectors as failed

    # --- Read Input Cell Image ---
    try:
        image = tiff.imread(input_cell_path)
    except FileNotFoundError:
        print(f"  Error: Input file not found at '{input_cell_path}'. Skipping.")
        return 0, expected_total_sectors
    except Exception as e:
        print(f"  Error reading TIFF file '{cell_filename}': {e}. Skipping.")
        return 0, expected_total_sectors

    # --- Validate Image & Determine Type ---
    if image.dtype != np.uint16:
        print(f"  Error: Image is not 16-bit grayscale (dtype: {image.dtype}). Skipping.")
        return 0, expected_total_sectors

    original_height, original_width = image.shape
    is_f_cell = cell_filename.lower().startswith('f')
    cell_type_description = "Unknown"

    # --- Verify dimensions based on filename prefix ---
    valid_dimensions = False
    if not is_f_cell:
        if original_width == 513 and original_height == 513:
            valid_dimensions = True
            cell_type_description = "Standard Cell (513x513)"
        else:
            print(f"  Error: Dimension mismatch for non-'f' cell '{cell_filename}'. "
                  f"Got {original_width}x{original_height}. Expected 513x513. Skipping.")
            return 0, expected_total_sectors
    else: # is_f_cell is True
        if original_width == 512 and original_height == 512:
            valid_dimensions = True
            cell_type_description = "Edge Cell (512x512, 'f')"
        elif original_width == 512 and original_height == 513:
             valid_dimensions = True
             cell_type_description = "Right Edge Cell (512x513, 'f')"
        elif original_width == 513 and original_height == 512:
             valid_dimensions = True
             cell_type_description = "Bottom Edge Cell (513x512, 'f')"
        else:
            print(f"  Error: Dimension mismatch for 'f' cell '{cell_filename}'. "
                  f"Got {original_width}x{original_height}. Expected 512x512, 512x513, or 513x512. Skipping.")
            return 0, expected_total_sectors

    if not valid_dimensions: # Should not happen if logic above is correct, but safety check
         print(f"  Internal Error: Dimension validation failed unexpectedly for {cell_filename}. Skipping.")
         return 0, expected_total_sectors

    print(f"  Type: {cell_type_description}")

    # Determine if overlap is missing on specific edges based on actual dimensions
    missing_right_overlap = (original_width == 512)
    missing_bottom_overlap = (original_height == 512)

    # --- Iterate Through Sector Grid (16x16) ---
    for r_idx in range(NUM_SECTORS_HIGH):  # Sector row index (0-15)
        for c_idx in range(NUM_SECTORS_WIDE): # Sector column index (0-15)

            # --- Calculate Base Slice Coordinates (Top-Left corner of BASE sector) ---
            slice_r_start = r_idx * BASE_SECTOR_HEIGHT # e.g., 0, 32, 64, ...
            slice_c_start = c_idx * BASE_SECTOR_WIDTH  # e.g., 0, 32, 64, ...

            # --- Determine Required Slice End (including desired overlap) ---
            # We always aim for the full overlap size initially
            slice_r_end_desired = slice_r_start + OUTPUT_SECTOR_HEIGHT # e.g., 33, 65, 97, ...
            slice_c_end_desired = slice_c_start + OUTPUT_SECTOR_WIDTH  # e.g., 33, 65, 97, ...

            # --- Calculate Actual Slice End (clipped to actual image boundaries) ---
            # This ensures we don't try to read past the image's real dimensions (513 or 512)
            slice_r_end_actual = min(slice_r_end_desired, original_height)
            slice_c_end_actual = min(slice_c_end_desired, original_width)

            # --- Determine if this specific OUTPUT sector needs an 'f' prefix ---
            # It needs 'f' if it falls on an edge where the INPUT cell was missing overlap
            needs_f_prefix_sector = False
            is_bottom_edge_sector = (r_idx == NUM_SECTORS_HIGH - 1)
            is_right_edge_sector = (c_idx == NUM_SECTORS_WIDE - 1)

            if is_right_edge_sector and missing_right_overlap:
                 needs_f_prefix_sector = True
            if is_bottom_edge_sector and missing_bottom_overlap:
                 needs_f_prefix_sector = True

            # --- Extract the Sector Data ---
            try:
                # Use the *actual* slice boundaries calculated above
                extracted_data = image[slice_r_start:slice_r_end_actual,
                                       slice_c_start:slice_c_end_actual]

                # --- Determine Output Sector Filename ---
                base_output_name = f"{r_idx:02d}_{c_idx:02d}.tif"
                output_filename = ("f" + base_output_name) if needs_f_prefix_sector else base_output_name
                output_path = os.path.join(output_subfolder, output_filename)

                # --- Validate Extracted Size (Optional but good sanity check) ---
                expected_h = OUTPUT_SECTOR_HEIGHT if not (is_bottom_edge_sector and missing_bottom_overlap) else BASE_SECTOR_HEIGHT
                expected_w = OUTPUT_SECTOR_WIDTH if not (is_right_edge_sector and missing_right_overlap) else BASE_SECTOR_WIDTH
                if extracted_data.shape != (expected_h, expected_w):
                    print(f"    Warning: Sector {output_filename} extracted size {extracted_data.shape} differs from expected {expected_h}x{expected_w}")


                # --- Save the Extracted Sector ---
                tiff.imwrite(output_path, extracted_data)
                # print(f"    Saved: {output_filename} (Size: {extracted_data.shape[1]}x{extracted_data.shape[0]})") # Verbose
                success_count += 1

            except IndexError as e:
                 sector_name_for_error = f"{r_idx:02d}_{c_idx:02d}"
                 print(f"  Error slicing sector {sector_name_for_error} for {cell_filename}: {e}. Check coordinates/dimensions.")
                 print(f"    Slice attempted: R={slice_r_start}:{slice_r_end_actual}, C={slice_c_start}:{slice_c_end_actual} on image {original_height}x{original_width}")
                 failure_count += 1
            except Exception as e:
                sector_name_for_error = f"{r_idx:02d}_{c_idx:02d}"
                print(f"  Error processing or saving sector {sector_name_for_error} for {cell_filename}: {e}")
                failure_count += 1

    print(f"  Finished processing '{cell_filename}'. Saved: {success_count} sectors, Failed: {failure_count} sectors.")
    # Return failure count based on difference from expected, ensuring total adds up
    calculated_failures = expected_total_sectors - success_count
    if calculated_failures != failure_count:
        print(f"  Warning: Mismatch in failure count logic ({failure_count} vs {calculated_failures}). Using calculated.")
    return success_count, calculated_failures


# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup ---
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # Create main output dir

    print("--- Starting Cell Splitting into Sectors ---")
    print(f"Input folder: '{INPUT_FOLDER}'")
    print(f"Output folder: '{OUTPUT_FOLDER}' (sectors will be in subfolders)")
    print(f"Base sector size (WxH): {BASE_SECTOR_WIDTH}x{BASE_SECTOR_HEIGHT}")
    print(f"Target sector size (WxH): Up to {OUTPUT_SECTOR_WIDTH}x{OUTPUT_SECTOR_HEIGHT} (with {SECTOR_OVERLAP}px overlap)")
    print(f"Expected sectors per cell: {NUM_SECTORS_WIDE}x{NUM_SECTORS_HIGH}")
    print(f"Handles standard 513x513 cells.")
    print(f"Handles edge cells prefixed 'f' with dimensions 512x512, 512x513, or 513x512.")
    print(f"Sectors on edges missing overlap (due to input cell size 512) will be prefixed 'f' and lack full overlap.")
    print("-" * 30)

    total_cells_processed = 0
    total_cells_skipped = 0
    total_sectors_saved = 0
    total_sectors_failed = 0
    expected_sectors_per_cell = NUM_SECTORS_WIDE * NUM_SECTORS_HIGH

    # --- Process Files ---
    processed_cell_count_actual = 0 # Count only those that pass initial checks
    for entry in os.listdir(INPUT_FOLDER):
        input_file_path = os.path.join(INPUT_FOLDER, entry)

        # Basic check for TIFF files
        if os.path.isfile(input_file_path) and (entry.lower().endswith('.tif') or entry.lower().endswith('.tiff')):
            total_cells_processed += 1 # Increment count of files *attempted*
            saved, failed = process_cell_image(input_file_path, OUTPUT_FOLDER)

            # Update totals
            total_sectors_saved += saved
            total_sectors_failed += failed

            # Check if the cell was skipped *inside* the function due to errors/validation
            if saved == 0 and failed == expected_sectors_per_cell:
                 total_cells_skipped += 1
            elif saved > 0 or failed > 0: # If any sectors were processed/failed within it
                 processed_cell_count_actual +=1


        elif os.path.isfile(input_file_path):
             # Might be other files in the input folder
             print(f"Skipping non-TIFF file: {entry}")
             total_cells_skipped += 1 # Count skipped non-TIFFs too


    print("-" * 30)
    print("--- Processing Summary ---")
    print(f"Input TIFF files found and attempted: {total_cells_processed}")
    # print(f"Input cell files processed (passed initial checks): {processed_cell_count_actual}") # Alternate metric
    print(f"Input files/cells skipped (non-TIFF, invalid format/size/error): {total_cells_skipped}")
    print(f"Total individual sectors saved successfully: {total_sectors_saved}")
    print(f"Total individual sector processing/saving failures: {total_sectors_failed}")
    # Sanity check: total saved + total failed should ideally equal processed_cells * sectors_per_cell
    expected_total_output_sectors = processed_cell_count_actual * expected_sectors_per_cell
    if total_sectors_saved + total_sectors_failed != expected_total_output_sectors:
         print(f"Warning: Sector count mismatch. Saved ({total_sectors_saved}) + Failed ({total_sectors_failed}) != Expected ({expected_total_output_sectors})")
         print("         This might occur if cells were skipped after partial processing or due to counting discrepancies.")

    print("--- Finished ---")