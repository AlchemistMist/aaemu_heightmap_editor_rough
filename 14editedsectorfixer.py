import numpy as np
import tifffile as tiff
import os
import sys
import shutil # For safe renaming

# --- Configuration ---
SECTORS_FOLDER = "edited_sectors"     # Input folder with 'f' files, will be modified
REFERENCE_FOLDER = "output_sectors"   # Folder containing the reference 'sector' files

# Expected non-edge sector size (for comparison)
EXPECTED_REF_WIDTH = 33
EXPECTED_REF_HEIGHT = 33
# --- End Configuration ---

def fix_single_f_sector(f_sector_path, reference_base_folder):
    """
    Attempts to fix a single 'f' prefixed sector file by adding overlap
    from its corresponding reference file located in the reference_base_folder.
    Modifies the file in place (in SECTORS_FOLDER) and renames it upon success.

    Args:
        f_sector_path (str): The full path to the 'f' prefixed sector file.
        reference_base_folder (str): Path to the base reference folder (e.g., "output_sectors").

    Returns:
        bool: True if successfully fixed and renamed, False otherwise.
    """
    print(f"  Attempting to fix: {os.path.basename(f_sector_path)}")
    f_dir_path = os.path.dirname(f_sector_path)            # e.g., edited_sectors/f001_002
    f_filename = os.path.basename(f_sector_path)           # e.g., f15_03.tif
    f_subfolder_name = os.path.basename(f_dir_path)        # e.g., f001_002

    # --- 1. Derive Reference and Target Paths ---
    try:
        if not f_filename.lower().startswith('f') or not f_filename.lower().endswith('.tif'):
            print(f"    Skipping: Not an 'f' prefixed .tif file.")
            return False

        # --- Derive Reference Info ---
        f_base_name_no_ext = os.path.splitext(f_filename)[0] # e.g., 'f15_03'
        numeric_part = f_base_name_no_ext[1:]                # e.g., '15_03'
        if not numeric_part:
             raise ValueError("Filename format invalid after removing 'f'")

        ref_filename = f"sector_{numeric_part}.tif"           # e.g., 'sector15_03.tif'

        # Derive reference subfolder name (remove leading 'f' if present)
        if f_subfolder_name.lower().startswith('f'):
            ref_subfolder_name = f_subfolder_name[1:]        # e.g., '001_002'
        else:
            # This case is less likely if previous steps worked, but handle it.
            print(f"    Warning: 'f' file found in non-'f' subfolder '{f_subfolder_name}'. Using this name for reference subfolder.")
            ref_subfolder_name = f_subfolder_name

        # *** Construct full path to the potential reference file USING REFERENCE_FOLDER ***
        ref_dir_path = os.path.join(reference_base_folder, ref_subfolder_name) # e.g., output_sectors/001_002
        ref_sector_path = os.path.join(ref_dir_path, ref_filename)             # e.g., output_sectors/001_002/sector15_03.tif

        # --- Derive Target Fixed Filename (for renaming) ---
        target_fixed_filename = f"{numeric_part}.tif"                          # e.g., '15_03.tif'
        target_fixed_path = os.path.join(f_dir_path, target_fixed_filename)    # e.g., edited_sectors/f001_002/15_03.tif (final path)

        print(f"    Looking for reference: {ref_sector_path}")

        if not os.path.exists(ref_sector_path):
            print(f"    Error: Reference file not found at '{ref_sector_path}'. Cannot fix.")
            return False

    except ValueError as ve:
         print(f"    Error parsing filename '{f_filename}': {ve}")
         return False
    except Exception as e:
        print(f"    Error deriving reference paths for '{f_filename}': {e}")
        return False

    # --- 2. Load Images ---
    # (Load logic remains the same)
    try:
        target_image = tiff.imread(f_sector_path)
        ref_image = tiff.imread(ref_sector_path)

        if target_image.dtype != np.uint16 or ref_image.dtype != np.uint16:
             print(f"    Error: One or both images are not uint16. Skipping.")
             return False

    except Exception as e:
        print(f"    Error reading target ('{f_filename}') or reference ('{ref_filename}') image: {e}")
        return False

    # --- 3. Determine Required Fixes and Prepare Canvas ---
    # (Logic remains the same)
    h_target, w_target = target_image.shape
    h_ref, w_ref = ref_image.shape

    missing_right = w_ref > w_target
    missing_bottom = h_ref > h_target

    if not missing_right and not missing_bottom:
        print(f"    Warning: Target file '{f_filename}' ({w_target}x{h_target}) "
              f"is not smaller than reference '{ref_filename}' ({w_ref}x{h_ref}). "
              f"Assuming already fixed or incorrect files. Renaming only.")
        combined_data = target_image # Use original data if no fix needed

    else:
        print(f"    Target size: {w_target}x{h_target}, Ref size: {w_ref}x{h_ref}. "
              f"Missing right: {missing_right}, Missing bottom: {missing_bottom}")

        try:
            new_canvas = np.zeros_like(ref_image) # Same size and dtype as reference
        except Exception as e:
            print(f"    Error creating numpy canvas: {e}")
            return False

        # --- 4. Combine Data ---
        # (Logic remains the same)
        try:
            copy_h = min(h_target, h_ref)
            copy_w = min(w_target, w_ref)
            new_canvas[:copy_h, :copy_w] = target_image[:copy_h, :copy_w]

            if missing_right:
                print(f"    Copying right overlap column from reference.")
                new_canvas[:h_ref, -1] = ref_image[:, -1]

            if missing_bottom:
                print(f"    Copying bottom overlap row from reference.")
                new_canvas[-1, :w_ref] = ref_image[-1, :]

            combined_data = new_canvas # Use the combined data

        except Exception as e:
            print(f"    Error combining image data: {e}")
            return False

    # --- 5. Save Combined Data Over Original 'f' File ---
    try:
        print(f"    Saving combined/original image back to: {f_sector_path}")
        tiff.imwrite(f_sector_path, combined_data)
    except Exception as e:
        print(f"    Error saving combined image to '{f_sector_path}': {e}")
        return False

    # --- 6. Rename the Fixed File ---
    if f_sector_path == target_fixed_path:
         print(f"    File '{f_filename}' already has the correct name ('{target_fixed_filename}'). Skipping rename.")
         return True # Considered success if already named correctly

    try:
        print(f"    Renaming '{f_filename}' to '{target_fixed_filename}'")
        shutil.move(f_sector_path, target_fixed_path)
        return True # Success!
    except Exception as e:
        print(f"    Error renaming file '{f_filename}' to '{target_fixed_filename}': {e}")
        return False # Renaming failed


# --- Main Execution ---
if __name__ == "__main__":
    # --- Initial Checks ---
    if not os.path.isdir(SECTORS_FOLDER):
        print(f"Error: Sectors folder '{SECTORS_FOLDER}' not found.")
        sys.exit(1)
    if not os.path.isdir(REFERENCE_FOLDER):
        print(f"Error: Reference folder '{REFERENCE_FOLDER}' not found.")
        sys.exit(1)

    print("--- Starting Edge Sector Fixing Process ---")
    print(f"Processing folder: '{SECTORS_FOLDER}'")
    print(f"Using reference folder: '{REFERENCE_FOLDER}'")
    print(f"Looking for 'f' prefixed files and their non-'f', 'sector' prefixed references in reference folder.")
    print("** This script modifies files and folders in '{SECTORS_FOLDER}' in place! Backup recommended. **")
    print("-" * 30)

    total_f_files_found = 0
    total_files_fixed = 0
    total_files_failed = 0
    total_dirs_renamed = 0
    total_dir_rename_fails = 0
    dirs_to_rename = {} # Store potential directory renames {old_path: new_path}

    # Use os.walk with topdown=False on the SECTORS_FOLDER
    for dirpath, dirnames, filenames in os.walk(SECTORS_FOLDER, topdown=False):
        current_subdir_name = os.path.basename(dirpath)
        dir_starts_with_f = current_subdir_name.lower().startswith('f')
        all_f_files_fixed_in_this_dir = True
        f_files_present_in_this_dir = False

        # Only process if it's a subdirectory (avoid processing SECTORS_FOLDER itself directly)
        if dirpath == SECTORS_FOLDER:
             continue

        print(f"\nScanning directory: {dirpath}")

        for filename in filenames:
            if filename.lower().startswith('f') and filename.lower().endswith('.tif'):
                f_files_present_in_this_dir = True
                total_f_files_found += 1
                file_path = os.path.join(dirpath, filename)

                # Pass the REFERENCE_FOLDER path to the fixing function
                if fix_single_f_sector(file_path, REFERENCE_FOLDER):
                    total_files_fixed += 1
                else:
                    total_files_failed += 1
                    all_f_files_fixed_in_this_dir = False

        # Check directory rename condition after processing files in it
        if dir_starts_with_f and f_files_present_in_this_dir:
            if all_f_files_fixed_in_this_dir:
                old_dir_path = dirpath
                new_dir_name = current_subdir_name[1:] # Remove 'f'
                parent_dir = os.path.dirname(old_dir_path)
                new_dir_path = os.path.join(parent_dir, new_dir_name)
                # Check if the target directory name already exists (e.g., if run twice)
                if os.path.exists(new_dir_path):
                    print(f"  Warning: Cannot schedule rename for '{current_subdir_name}' because '{new_dir_name}' already exists. Skipping rename.")
                else:
                    dirs_to_rename[old_dir_path] = new_dir_path
                    print(f"  Directory '{current_subdir_name}' is fully fixed, scheduled for rename.")
            else:
                print(f"  Directory '{current_subdir_name}' still contains unfixed/failed 'f' files. Skipping rename.")
        elif dir_starts_with_f and not f_files_present_in_this_dir:
             print(f"  Directory '{current_subdir_name}' starts with 'f' but contains no 'f' files. Consider manual check/rename.")


    # --- Perform Directory Renames After Traversal ---
    print("\n--- Performing Directory Renames ---")
    if not dirs_to_rename:
        print("No directories marked for renaming.")
    else:
        # Sort keys might help prevent issues if paths are nested, though less likely here
        for old_dir_path in sorted(dirs_to_rename.keys()):
             new_dir_path = dirs_to_rename[old_dir_path]
             if old_dir_path == new_dir_path: continue # Safety check
             if not os.path.exists(old_dir_path):
                  print(f"Skipping rename: Source directory '{os.path.basename(old_dir_path)}' no longer exists.")
                  continue
             if os.path.exists(new_dir_path):
                  print(f"Error: Cannot rename '{os.path.basename(old_dir_path)}' to '{os.path.basename(new_dir_path)}' because destination already exists!")
                  total_dir_rename_fails += 1
                  continue
             try:
                  print(f"Renaming directory '{os.path.basename(old_dir_path)}' to '{os.path.basename(new_dir_path)}'")
                  shutil.move(old_dir_path, new_dir_path)
                  total_dirs_renamed += 1
             except Exception as e:
                  print(f"Error renaming directory '{old_dir_path}': {e}")
                  total_dir_rename_fails += 1


    print("-" * 30)
    print("--- Fixing Process Summary ---")
    print(f"Target folder processed: '{SECTORS_FOLDER}'")
    print(f"Reference folder used: '{REFERENCE_FOLDER}'")
    print(f"Total 'f' prefixed sector files found: {total_f_files_found}")
    print(f"Successfully fixed and renamed: {total_files_fixed}")
    print(f"Failed to fix or rename: {total_files_failed}")
    print(f"Directories successfully renamed (f removed): {total_dirs_renamed}")
    print(f"Directory rename failures: {total_dir_rename_fails}")
    print("--- Finished ---")