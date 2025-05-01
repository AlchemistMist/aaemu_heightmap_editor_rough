import numpy as np
import tifffile as tiff
import os
import sys
import re # Regular expressions for robust filename parsing

# --- Configuration ---
INPUT_FOLDER = "edited_tiff"
OUTPUT_FOLDER = "edited_cells"

# Cell dimensions (base size, output includes overlap)
BASE_CELL_WIDTH = 512
BASE_CELL_HEIGHT = 512
OVERLAP = 1 # Pixels to overlap downwards and rightwards
OUTPUT_CELL_WIDTH = BASE_CELL_WIDTH + OVERLAP
OUTPUT_CELL_HEIGHT = BASE_CELL_HEIGHT + OVERLAP

# Filename pattern (adjust if needed, e.g., different separators)
# Matches XXX_YYY_XXX_YYY format where X and Y are digits
FILENAME_PATTERN = re.compile(r"^(\d{3})_(\d{3})_(\d{3})_(\d{3})\.tif[f]?$", re.IGNORECASE)

# --- End Configuration ---


def parse_input_filename(filename):
    """
    Parses filename like '000_000_003_003.tif' into grid coordinates.

    Args:
        filename (str): The input filename.

    Returns:
        tuple: (start_row, start_col, end_row, end_col) as integers,
               or None if the format is invalid.
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None # Doesn't match the expected format

    try:
        start_row = int(match.group(1))
        start_col = int(match.group(2))
        end_row = int(match.group(3))
        end_col = int(match.group(4))
        return start_row, start_col, end_row, end_col
    except (ValueError, IndexError):
        return None # Should not happen with regex match, but safety first


def split_image_into_cells(input_path, output_folder):
    """
    Splits a large TIFF image into smaller cells based on its filename,
    adding overlap and handling edge cases.

    Args:
        input_path (str): Path to the large input TIFF file.
        output_folder (str): Path to the directory to save cell files.

    Returns:
        tuple: (success_count, failure_count) for cells from this image.
    """
    print(f"\nProcessing: {os.path.basename(input_path)}")
    filename = os.path.basename(input_path)
    success_count = 0
    failure_count = 0

    # --- 1. Parse Filename for Grid Info ---
    grid_coords = parse_input_filename(filename)
    if grid_coords is None:
        print(f"  Error: Invalid filename format. Cannot determine grid. Skipping.")
        return 0, 1 # 0 success, 1 failure (the whole file)

    start_row, start_col, end_row, end_col = grid_coords
    print(f"  Grid definition from filename: Rows {start_row}-{end_row}, Cols {start_col}-{end_col}")

    num_cells_high = end_row - start_row + 1
    num_cells_wide = end_col - start_col + 1

    if num_cells_high <= 0 or num_cells_wide <= 0:
         print(f"  Error: Invalid grid range in filename (start > end). Skipping.")
         return 0, 1

    # --- 2. Read Input Image ---
    try:
        image = tiff.imread(input_path)
    except FileNotFoundError:
        print(f"  Error: Input file not found at '{input_path}'. Skipping.")
        return 0, 1
    except Exception as e:
        print(f"  Error reading TIFF file '{filename}': {e}. Skipping.")
        return 0, 1

    # --- 3. Validate Image ---
    if image.dtype != np.uint16:
        print(f"  Error: Image is not 16-bit grayscale (dtype: {image.dtype}). Skipping.")
        return 0, 1

    original_height, original_width = image.shape
    expected_height = num_cells_high * BASE_CELL_HEIGHT
    expected_width = num_cells_wide * BASE_CELL_WIDTH

    if original_height != expected_height or original_width != expected_width:
        print(f"  Error: Image dimensions ({original_width}x{original_height}) do not match "
              f"expected size ({expected_width}x{expected_height}) based on filename grid and "
              f"BASE cell size ({BASE_CELL_WIDTH}x{BASE_CELL_HEIGHT}). Skipping.")
        return 0, 1

    print(f"  Input image validated: {original_width}x{original_height}, uint16.")

    # --- 4. Iterate Through Cell Grid and Extract/Save ---
    for r_idx in range(num_cells_high):  # Index within the grid (0 to num_cells_high-1)
        for c_idx in range(num_cells_wide): # Index within the grid (0 to num_cells_wide-1)

            current_cell_row = start_row + r_idx
            current_cell_col = start_col + c_idx

            # --- Calculate Slice Coordinates (Top-Left corner of BASE cell) ---
            slice_r_start = r_idx * BASE_CELL_HEIGHT
            slice_c_start = c_idx * BASE_CELL_WIDTH

            # --- Determine Required Slice End (including overlap) ---
            # These are the *desired* end points, might exceed image bounds
            slice_r_end_desired = slice_r_start + OUTPUT_CELL_HEIGHT
            slice_c_end_desired = slice_c_start + OUTPUT_CELL_WIDTH

            # --- Check for Edge Cases (will we go out of bounds?) ---
            is_bottom_edge = slice_r_end_desired > original_height
            is_right_edge = slice_c_end_desired > original_width
            needs_f_prefix = is_bottom_edge or is_right_edge

            # --- Calculate Actual Slice End (clipped to image boundaries) ---
            slice_r_end_actual = min(slice_r_end_desired, original_height)
            slice_c_end_actual = min(slice_c_end_desired, original_width)

            # --- Extract the Cell Data (potentially smaller than 513x513 on edges) ---
            try:
                # Use the *actual* slice boundaries clipped to the image size
                extracted_data = image[slice_r_start:slice_r_end_actual,
                                       slice_c_start:slice_c_end_actual]

                # --- Determine Output Filename ---
                base_output_name = f"{current_cell_row:03d}_{current_cell_col:03d}.tif"
                output_filename = ("f" + base_output_name) if needs_f_prefix else base_output_name
                output_path = os.path.join(output_folder, output_filename)

                # --- Save the Extracted Cell ---
                tiff.imwrite(output_path, extracted_data)
                # print(f"    Saved: {output_filename} (Size: {extracted_data.shape[1]}x{extracted_data.shape[0]})") # Verbose log
                success_count += 1

            except Exception as e:
                print(f"  Error processing or saving cell {current_cell_row:03d}_{current_cell_col:03d}: {e}")
                failure_count += 1

    print(f"  Finished processing '{filename}'. Saved: {success_count} cells, Failed: {failure_count} cells.")
    return success_count, failure_count


# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup ---
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("--- Starting Image Cell Splitting ---")
    print(f"Input folder: '{INPUT_FOLDER}'")
    print(f"Output folder: '{OUTPUT_FOLDER}'")
    print(f"Base cell size (WxH): {BASE_CELL_WIDTH}x{BASE_CELL_HEIGHT}")
    print(f"Output cell size (WxH): Up to {OUTPUT_CELL_WIDTH}x{OUTPUT_CELL_HEIGHT} (with {OVERLAP}px overlap)")
    print(f"Edge cells (without full overlap) will be prefixed with 'f'.")
    print("-" * 30)

    total_files_processed = 0
    total_cells_saved = 0
    total_cells_failed = 0
    skipped_files = 0

    # --- Process Files ---
    for entry in os.listdir(INPUT_FOLDER):
        input_file_path = os.path.join(INPUT_FOLDER, entry)

        # Check if it's a file and looks like a potential candidate
        if os.path.isfile(input_file_path) and FILENAME_PATTERN.match(entry):
            total_files_processed += 1
            saved, failed = split_image_into_cells(input_file_path, OUTPUT_FOLDER)
            total_cells_saved += saved
            # A file skip counts as one failure in this context, handled inside function
            if saved == 0 and failed > 0:
                 skipped_files += 1 # Count files that were skipped entirely
                 total_cells_failed += failed # Add potential internal cell failures too

            # Update total cell failures if some cells failed within a processed file
            elif failed > 0 :
                 total_cells_failed += failed

        elif os.path.isfile(input_file_path) and (entry.lower().endswith('.tif') or entry.lower().endswith('.tiff')):
             # It's a TIFF file but doesn't match the expected naming pattern
             print(f"Skipping file with unexpected name format: {entry}")
             skipped_files += 1


    print("-" * 30)
    print("--- Processing Summary ---")
    print(f"Input files processed or attempted: {total_files_processed}")
    print(f"Input files skipped (invalid name/format/error): {skipped_files}")
    print(f"Total individual cells saved successfully: {total_cells_saved}")
    print(f"Total individual cell failures (read/write/process): {total_cells_failed}")
    print("--- Finished ---")