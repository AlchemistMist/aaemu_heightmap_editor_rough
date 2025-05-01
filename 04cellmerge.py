import os
import sys
import numpy as np
import tifffile # Preferred for scientific TIFFs (like 16-bit)

# --- Configuration ---
INPUT_DIR = "sector_merged"       # Folder containing the XXX_YYY.tif cell images
OUTPUT_DIR = "raw_input"        # Folder where the final merged map will be saved

# -- Define the area to merge --
# Cell names are inclusive (e.g., 000_000 to 008_008 means a 9x9 grid)
START_CELL = "020_030"
END_CELL   = "021_031"
# --- End Configuration ---


def parse_cell_name(cell_name):
    """Parses 'XXX_YYY' string into integer (row, col) tuple."""
    try:
        parts = cell_name.split('_')
        if len(parts) != 2:
            raise ValueError("Name must be in 'XXX_YYY' format")
        row = int(parts[0])
        col = int(parts[1])
        return row, col
    except (ValueError, TypeError, IndexError) as e:
        print(f"Error: Invalid cell name format '{cell_name}'. {e}")
        return None

def merge_cells(input_dir, output_dir, start_cell_str, end_cell_str):
    """
    Merges cell images within the specified range into a single large TIFF.
    """
    print("--- Starting Cell Merge ---")
    print(f"Input directory: '{input_dir}'")
    print(f"Output directory: '{output_dir}'")
    print(f"Merging from '{start_cell_str}' to '{end_cell_str}'")

    # --- 1. Validate Inputs and Parse Coordinates ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return False

    start_coords = parse_cell_name(start_cell_str)
    end_coords = parse_cell_name(end_cell_str)

    if start_coords is None or end_coords is None:
        return False # Error already printed by parse_cell_name

    start_row, start_col = start_coords
    end_row, end_col = end_coords

    if start_row > end_row or start_col > end_col:
        print(f"Error: Start cell ({start_row},{start_col}) must be less than or equal to End cell ({end_row},{end_col}).")
        return False

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return False

    # --- 2. Determine Grid and Cell Dimensions ---
    num_cells_high = end_row - start_row + 1
    num_cells_wide = end_col - start_col + 1
    print(f"Grid size: {num_cells_wide} cells wide x {num_cells_high} cells high.")

    cell_height = -1
    cell_width = -1
    cell_dtype = None
    first_cell_found = False

    # Find the *first* expected cell file that *actually* exists to get dimensions
    first_cell_path = None
    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            test_filename = f"{r:03d}_{c:03d}.tif"
            test_filepath = os.path.join(input_dir, test_filename)
            if os.path.exists(test_filepath):
                try:
                    with tifffile.TiffFile(test_filepath) as tif:
                        # Basic check, assumes first series/page is the image
                        page = tif.pages[0]
                        cell_shape = page.shape
                        cell_dtype = page.dtype
                        if len(cell_shape) == 2: # Expecting 2D grayscale
                             cell_height, cell_width = cell_shape
                        else:
                             print(f"Warning: First found cell '{test_filename}' has unexpected shape {cell_shape}. Assuming 2D.")
                             # Attempt to handle common cases (e.g., extra singleton dimension)
                             if len(cell_shape) == 3 and cell_shape[0] == 1:
                                 cell_height, cell_width = cell_shape[1], cell_shape[2]
                             elif len(cell_shape) == 3 and cell_shape[2] == 1:
                                  cell_height, cell_width = cell_shape[0], cell_shape[1]
                             else:
                                 raise ValueError(f"Unsupported shape {cell_shape}")

                        if cell_dtype != np.uint16:
                            print(f"Warning: Cell data type is {cell_dtype}, expected uint16. Proceeding, but output will be uint16.")
                            cell_dtype = np.uint16 # Force output to uint16

                        print(f"Detected cell dimensions from '{test_filename}': {cell_width}x{cell_height}, dtype: {cell_dtype}")
                        first_cell_path = test_filepath # Save path for potential reload if needed
                        first_cell_found = True
                        break # Stop searching once dimensions are found
                except Exception as e:
                    print(f"Error reading dimensions from first found cell '{test_filepath}': {e}")
                    return False # Cannot proceed without knowing dimensions
        if first_cell_found:
            break

    if not first_cell_found:
        print(f"Error: Could not find any existing cell files in the range '{start_cell_str}' to '{end_cell_str}' in '{input_dir}' to determine dimensions.")
        return False

    # --- 3. Create Output Canvas ---
    total_height = num_cells_high * cell_height
    total_width = num_cells_wide * cell_width
    print(f"Total output size: {total_width}x{total_height} pixels.")

    try:
        # Initialize with zeros (black for grayscale)
        # Ensure the dtype matches the determined (or forced) cell_dtype
        merged_canvas = np.zeros((total_height, total_width), dtype=cell_dtype)
        print("Created output canvas in memory.")
    except MemoryError:
        print(f"Error: Insufficient memory to create canvas of size {total_width}x{total_height} ({total_height*total_width*cell_dtype.itemsize / (1024**3):.2f} GB).")
        print("Consider processing a smaller range or using a system with more RAM.")
        return False
    except Exception as e:
        print(f"Error creating numpy canvas: {e}")
        return False


    # --- 4. Load and Place Each Cell ---
    missing_files = []
    dimension_mismatches = []
    processed_count = 0

    for current_row in range(start_row, end_row + 1):
        for current_col in range(start_col, end_col + 1):
            cell_filename = f"{current_row:03d}_{current_col:03d}.tif"
            cell_filepath = os.path.join(input_dir, cell_filename)

            # Calculate placement position (relative to the top-left cell in the range)
            # y_idx is the row index within our output grid (0 to num_cells_high-1)
            # x_idx is the column index within our output grid (0 to num_cells_wide-1)
            y_idx = current_row - start_row
            x_idx = current_col - start_col

            y_offset = y_idx * cell_height
            x_offset = x_idx * cell_width

            if os.path.exists(cell_filepath):
                try:
                    # print(f"  Processing {cell_filename}...") # Verbose log
                    cell_data = tifffile.imread(cell_filepath)

                    # Validate dimensions and type
                    if cell_data.shape != (cell_height, cell_width):
                        print(f"  Warning: Dimension mismatch for '{cell_filename}'. Expected {cell_height}x{cell_width}, got {cell_data.shape}. Skipping placement.")
                        dimension_mismatches.append(cell_filename)
                        continue # Skip this cell

                    if cell_data.dtype != cell_dtype:
                         # This might happen if the first cell check didn't catch variation
                         # Or if we forced uint16 earlier
                         # print(f"  Warning: Data type mismatch for '{cell_filename}'. Expected {cell_dtype}, got {cell_data.dtype}. Attempting conversion.")
                         try:
                             cell_data = cell_data.astype(cell_dtype)
                         except Exception as e_conv:
                              print(f"    Error converting dtype for '{cell_filename}': {e_conv}. Skipping placement.")
                              continue


                    # Place onto canvas
                    merged_canvas[y_offset : y_offset + cell_height,
                                  x_offset : x_offset + cell_width] = cell_data
                    processed_count += 1

                except MemoryError:
                     print(f"Error: Ran out of memory while reading '{cell_filename}'. Aborting.")
                     # Optionally try to save partial result? Difficult.
                     return False
                except Exception as e:
                    print(f"  Error reading or processing '{cell_filename}': {e}. Skipping placement.")
                    missing_files.append(f"{cell_filename} (Read Error: {e})") # Treat read errors like missing
            else:
                print(f"  Warning: Cell file not found: '{cell_filename}'. Leaving corresponding area black.")
                missing_files.append(cell_filename)
                # The area remains black (zeros) by default

    print(f"\nProcessed {processed_count} cell files.")
    if missing_files:
        print("Missing or error reading the following cell files:")
        for mf in missing_files: print(f"  - {mf}")
    if dimension_mismatches:
        print("The following cell files had incorrect dimensions and were skipped:")
        for dm in dimension_mismatches: print(f"  - {dm}")

    # --- 5. Save Final Merged Image ---
    if processed_count == 0 and not missing_files:
         print("\nNo cell files were found or processed successfully. Nothing to save.")
         return True # Not an error, just nothing done

    output_filename = f"{start_cell_str}_{end_cell_str}.tif"
    output_filepath = os.path.join(output_dir, output_filename)

    print(f"\nSaving final merged image to '{output_filepath}'...")
    try:
        # Use tifffile to save, good for large files and specific dtypes
        tifffile.imwrite(output_filepath, merged_canvas, imagej=False) # imagej=True adds extra metadata if needed
        print("Save complete.")
        return True
    except Exception as e:
        print(f"Error saving final image: {e}")
        return False


# --- Main execution ---
if __name__ == "__main__":
    success = merge_cells(INPUT_DIR, OUTPUT_DIR, START_CELL, END_CELL)

    if success:
        print("\n--- Cell Merge Finished Successfully ---")
    else:
        print("\n--- Cell Merge Failed ---")
        sys.exit(1) # Exit with error code if failed