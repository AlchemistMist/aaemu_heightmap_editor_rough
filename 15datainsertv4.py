import struct
import os
import math
import numpy as np
import tifffile
import cv2 # For resizing
import shutil # For copying files

# --- Configuration ---
EDITED_SECTORS_BASE_DIR = "edited_sectors" # Folder containing XXX_YYY subfolders with sector TIFs
ORIGINAL_DATA_BASE_DIR = "."  # Assumes original folders (e.g., 000_000) are in the script's dir
OUTPUT_BASE_DIR = "edited_heightmap_dynamic" # Renamed output dir
MIN_SECTOR_INDEX = 0
MAX_SECTOR_INDEX = 15

# Assumption: Input TIFF files contain float32 values representing actual world heights.
# The values might range from 0 to 2048 globally, but each TIF's actual min/max will be used.
INPUT_TIFF_ASSUMED_DTYPE = np.float32

# --- Height Configuration (REMOVED - Now dynamic per sector) ---
# terrainstartheight = 0.0
imageheight = 2048.0
# terrainscaling = ...

# Define the search patterns (identifying metadata blocks)
SEARCH_STRINGS = [
    bytes([0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), # 33x33 res + flags/id?
    bytes([0x21, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00]),
    bytes([0x09, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]), # 9x9
    bytes([0x11, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]), # 17x17
    bytes([0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]), # 5x5
    bytes([0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]), # 2x2
    bytes([0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]), # 3x3
    bytes([0x11, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]), # 17x17 var 2?
    bytes([0x11, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00]), # 17x17 var 3?
    bytes([0x21, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00]), # 33x33 var 9?
    bytes([0x21, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00]), # 33x33 var A?
    bytes([0x21, 0x00, 0x00, 0x00, 0x0B, 0x00, 0x00, 0x00]), # 33x33 var B?
    bytes([0x21, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00]), # 33x33 var C?
    bytes([0x21, 0x00, 0x00, 0x00, 0x0D, 0x00, 0x00, 0x00]), # 33x33 var D?
    bytes([0x21, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00]), # 33x33 var E?
    bytes([0x21, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00]), # 33x33 var F?
    bytes([0x21, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00])  # 33x33 var 10?
]
VALID_RESOLUTIONS = [33, 17, 9, 5, 3, 2] # Resolutions expected in heightmap.dat
# --- End Configuration ---


# --- Helper Functions ---
def read_float32(data, offset):
    """Reads a 32-bit little-endian float from bytes."""
    if offset < 0 or offset + 4 > len(data):
        raise IndexError(f"Offset out of bounds for reading float32: offset={offset}, len={len(data)}")
    return struct.unpack('<f', data[offset:offset+4])[0]

def read_uint32(data, offset):
    """Reads a 32-bit little-endian unsigned integer from bytes."""
    if offset < 0 or offset + 4 > len(data):
        raise IndexError(f"Offset out of bounds for reading uint32: offset={offset}, len={len(data)}")
    return struct.unpack('<I', data[offset:offset+4])[0]

def write_float32(file_handle, offset, value):
    """Writes a 32-bit little-endian float to the file at the specified offset."""
    # Ensure value is float before packing
    try:
        packed_value = struct.pack('<f', float(value))
        file_handle.seek(offset)
        file_handle.write(packed_value)
    except OverflowError:
         print(f"  Error: OverflowError packing float value {value} at offset {offset}. Check range.")
         raise # Re-raise the error to stop processing this file
    except Exception as e:
         print(f"  Error writing float {value} at offset {offset}: {e}")
         raise # Re-raise

def check_sector(startx, starty, firstStartX, firstStartY):
    """Calculate sectors based on given coordinates."""
    try:
        sector_size = 64 # Assumed fixed sector size for coordinate calculation
        if sector_size == 0:
             print("Error: sector_size is zero in check_sector.")
             return None, None
        x = math.floor((float(startx) - float(firstStartX)) / sector_size)
        y = math.floor((float(starty) - float(firstStartY)) / sector_size)
        return x, y
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"Error calculating sector for ({startx}, {starty}) with base ({firstStartX}, {firstStartY}): {e}")
        return None, None
# --- End Helper Functions ---

def load_and_prepare_image(tiff_path, target_resolution):
    """
    Loads a TIFF image, ensuring it's float32, and resizes if its dimensions
    do not match the target_resolution needed for the heightmap.dat block.

    Returns:
        Numpy array (float32) or None if loading/conversion fails.
    """
    try:
        image_data = tifffile.imread(tiff_path)
        print(f"    Loaded TIFF: {os.path.basename(tiff_path)} (Shape: {image_data.shape}, Dtype: {image_data.dtype})")
    except FileNotFoundError:
        print(f"  Error: Input TIFF file not found: {tiff_path}")
        return None
    except Exception as e:
        print(f"  Error reading TIFF file {tiff_path}: {e}")
        return None

    # --- Ensure Float32 Dtype ---
    # Convert integers or other float types to float32 for consistent processing
    if image_data.dtype != np.float32:
        print(f"    Info: Converting image data from {image_data.dtype} to float32.")
        try:
            image_data = image_data.astype(np.float32)
        except Exception as e:
             print(f"  Error converting image data to float32: {e}. Skipping image.")
             return None

    # --- Check and Resize if Necessary ---
    current_shape = (image_data.shape[0], image_data.shape[1]) # Height x Width
    target_shape = (target_resolution, target_resolution)

    if current_shape != target_shape:
        print(f"  Info: Resizing image {os.path.basename(tiff_path)} from {current_shape} to {target_shape}.")
        # INTER_LINEAR is generally preferred for height data resizing
        resized_image = cv2.resize(image_data, (target_resolution, target_resolution), interpolation=cv2.INTER_LINEAR)

        # Ensure resizing didn't change the dtype unexpectedly (it shouldn't for float32)
        if resized_image.dtype != np.float32:
             print("    Warning: Resized image dtype changed unexpectedly. Attempting conversion back to float32.")
             resized_image = resized_image.astype(np.float32) # Force back if needed
        return resized_image
    else:
        # No resizing needed
        print(f"    Image resolution {current_shape} matches target {target_shape}. No resizing needed.")
        return image_data


def transform_image_to_raw_dynamic(image_data_float, min_val_float, max_val_float):
    """
    Transforms float image data (representing actual heights) into raw uint16 data (0-65535).
    The transformation maps the image's actual min_val to raw 0 and max_val to raw 65535.

    Args:
        image_data_float (np.ndarray): Numpy array of float32 heights.
        min_val_float (float): The minimum height value found in this image data.
        max_val_float (float): The maximum height value found in this image data.

    Returns:
        np.ndarray: Numpy array of uint16 raw values.
    """
    height_range = max_val_float - min_val_float

    print(f"    Transforming data: Img Min={min_val_float:.4f}, Img Max={max_val_float:.4f}, Range={height_range:.4f}")

    # Handle flat terrain (min == max)
    # Use a small tolerance for floating point comparison
    if height_range <= 1e-6:
        print("    Info: Flat terrain detected (min approx equal to max). Mapping all raw values to 0.")
        # All raw values should correspond to the min_val, which maps to raw 0.
        # The scaling value will also be 0, so the engine should interpret these 0s
        # as the sector_start_height.
        return np.zeros_like(image_data_float, dtype=np.uint16)
    else:
        # Apply the formula: raw = ((pixel - min) / range) * 65535
        # Use float64 for intermediate calculation for better precision, especially division
        image_data_f64 = image_data_float.astype(np.float64)
        min_f64 = np.float64(min_val_float)
        range_f64 = np.float64(height_range)

        raw_float = ((image_data_f64 - min_f64) / range_f64) * 65535.0

        # Clip to ensure values are within [0, 65535], round, and convert to uint16
        # Clipping handles potential minor floating point inaccuracies near the edges
        clipped_raw = np.clip(raw_float, 0.0, 65535.0)
        rounded_raw = np.round(clipped_raw)
        final_raw_uint16 = rounded_raw.astype(np.uint16)

        # Optional: Verify min/max of output (should be close to 0 and 65535 if range wasn't tiny)
        # print(f"    Transformed Raw Min={np.min(final_raw_uint16)}, Raw Max={np.max(final_raw_uint16)}")
        return final_raw_uint16


def update_heightmap_file_dynamic(subfolder_name):
    """
    Processes a heightmap.dat file: finds sectors, calculates dynamic start height
    and scaling based on TIFF min/max, loads TIFF, resizes, transforms TIFF data
    to raw uint16 based on dynamic range, and writes params & transformed data.
    """
    print(f"\nProcessing file in folder: {subfolder_name}")
    original_heightmap_path = os.path.join(ORIGINAL_DATA_BASE_DIR, subfolder_name, "client", "terrain", "heightmap.dat")
    output_heightmap_path = os.path.join(OUTPUT_BASE_DIR, subfolder_name, "client", "terrain", "heightmap.dat")
    edited_sector_images_dir = os.path.join(EDITED_SECTORS_BASE_DIR, subfolder_name) # Path to TIFs

    # 1. Check original heightmap
    if not os.path.exists(original_heightmap_path):
        print(f"  Skipping: Original heightmap.dat not found at {original_heightmap_path}")
        return 0

    # 2. Check if the directory containing edited TIFs exists
    if not os.path.isdir(edited_sector_images_dir):
        print(f"  Skipping: Edited sectors image directory not found at {edited_sector_images_dir}")
        return 0

    # 3. Prepare output directory and copy original file
    try:
        os.makedirs(os.path.dirname(output_heightmap_path), exist_ok=True)
        shutil.copy2(original_heightmap_path, output_heightmap_path)
        print(f"  Copied original to {output_heightmap_path}")
    except Exception as e:
        print(f"  Error preparing output file/directory: {e}")
        return 0

    updated_sectors_count = 0
    file_updated = False

    # 4. Open the *copied* file in read-write binary mode
    try:
        with open(output_heightmap_path, "rb+") as f:
            file_data = f.read()
            data_length = len(file_data)
            search_string_length = 8

            if data_length < 172:
                 print(f"  Warning: File {output_heightmap_path} is too small ({data_length} bytes). Skipping.")
                 return 0

            # --- Find base coordinates ---
            try:
                first_start_x = read_float32(file_data, 164)
                first_start_y = read_float32(file_data, 168)
                print(f"  Base coordinates found: X={first_start_x}, Y={first_start_y}")
            except IndexError:
                print("  Error: Could not read base coordinates (offset 164/168). Cannot calculate indices.")
                return 0
            # --- End finding base coords ---

            # 5. Iterate through file data searching for blocks
            current_offset = 0
            while current_offset <= data_length - search_string_length:
                found_match = False
                for pattern in SEARCH_STRINGS:
                    if file_data[current_offset : current_offset + search_string_length] == pattern:
                        found_match = True
                        match_offset = current_offset

                        block_start_offset = match_offset - 37
                        if block_start_offset < 0:
                            print(f"  Warning: Found pattern match at offset {match_offset}, block start offset {block_start_offset} invalid. Skipping.")
                            current_offset = match_offset + 1
                            found_match = False
                            break

                        try:
                            # --- Read Metadata ---
                            map_string_val = read_uint32(file_data, match_offset) # Resolution for this block
                            start_x = read_float32(file_data, block_start_offset + 4)
                            start_y = read_float32(file_data, block_start_offset + 8)
                            # --- End Read Metadata ---

                            # --- Calculate Sector Index ---
                            sector_x, sector_y = check_sector(start_x, start_y, first_start_x, first_start_y)
                            if sector_x is None:
                                print(f"    Skipping block at offset {block_start_offset} due to sector calculation error.")
                                current_offset = match_offset + 1
                                found_match = False
                                break

                            # --- FILTERING: Check if sector is in target range ---
                            if not (MIN_SECTOR_INDEX <= sector_x <= MAX_SECTOR_INDEX and MIN_SECTOR_INDEX <= sector_y <= MAX_SECTOR_INDEX):
                                current_offset = match_offset + 1
                                found_match = False
                                break

                            # --- Sector is valid and in range ---
                            target_resolution = map_string_val
                            if target_resolution not in VALID_RESOLUTIONS:
                                print(f"  Warning: Sector {sector_x:02d}_{sector_y:02d} has unexpected resolution {target_resolution} at match offset {match_offset}. Skipping.")
                                current_offset = match_offset + 1
                                found_match = False
                                break

                            # Construct expected input TIFF filename
                            tiff_filename = f"{sector_x:02d}_{sector_y:02d}.tif"
                            tiff_filepath = os.path.join(edited_sector_images_dir, tiff_filename)

                            print(f"  Processing sector {sector_x:02d}_{sector_y:02d} (Res: {target_resolution}x{target_resolution}) - Block Offset: {block_start_offset}, Match Offset: {match_offset}")

                            # --- Load, ensure float32, potentially resize ---
                            image_data_float = load_and_prepare_image(tiff_filepath, target_resolution)

                            if image_data_float is None:
                                print(f"    Skipping update for sector {sector_x:02d}_{sector_y:02d} due to image loading/processing error.")
                                current_offset = match_offset + 1
                                found_match = False
                                break

                            # --- DYNAMICALLY Calculate Start Height and Scaling ---
                            
                            imagescale = float(imageheight) / 65535.0
                            image_min_val = np.min(image_data_float)
                            image_max_val = np.max(image_data_float)

                            sector_start_height = float(image_min_val * imagescale) # Use the actual minimum value
                            height_range = float(image_max_val * imagescale) - float(image_min_val *imagescale)

                            if height_range > 1e-6: # Use tolerance for float comparison
                                sector_scaling = height_range / 65535.0
                            else:
                                sector_scaling = 0.0 # Scaling is 0 for flat terrain

                            print(f"    Dynamic Params: StartHeight={sector_start_height:.4f}, Scaling={sector_scaling:.8f} (based on image range {image_min_val:.4f} to {image_max_val:.4f})")
                            # --- End Dynamic Calculation ---

                            # --- Transform image data based on its own min/max ---
                            raw_image_data_uint16 = transform_image_to_raw_dynamic(image_data_float, image_min_val, image_max_val)

                            # Convert transformed numpy array to bytes
                            image_data_bytes = raw_image_data_uint16.tobytes()
                            bytes_per_pixel = 2
                            expected_data_size = target_resolution * target_resolution * bytes_per_pixel

                            # --- Validate Size ---
                            if len(image_data_bytes) != expected_data_size:
                                 print(f"  Error: Size mismatch for transformed data sector {sector_x:02d}_{sector_y:02d}. Expected {expected_data_size}, got {len(image_data_bytes)}. Skipping write.")
                                 current_offset = match_offset + 1
                                 found_match = False
                                 break

                            # --- Calculate write offsets ---
                            image_data_start_offset = match_offset + 8
                            start_height_offset = match_offset - 8
                            scaling_offset = match_offset - 4

                            # --- Sanity checks before writing ---
                            if start_height_offset < 0 or scaling_offset < 0:
                                print(f"  Error: Calculated negative offset for height/scaling write at match offset {match_offset}. Skipping.")
                                current_offset = match_offset + 1
                                found_match = False
                                break
                            if image_data_start_offset + expected_data_size > data_length:
                                print(f"  Error: Calculated image write position for sector {sector_x:02d}_{sector_y:02d} goes beyond file length. Skipping.")
                                current_offset = match_offset + 1
                                found_match = False
                                break
                            if start_height_offset + 4 > data_length or scaling_offset + 4 > data_length:
                                 print(f"  Error: Calculated param write position at {match_offset} goes beyond file length. Skipping.")
                                 current_offset = match_offset + 1
                                 found_match = False
                                 break

                            # --- Perform Writes (Dynamic Params, Transformed Data) ---
                            print(f"    Writing DYNAMIC terrain start height ({sector_start_height:.4f}) to offset {start_height_offset}")
                            write_float32(f, start_height_offset, sector_start_height)

                            print(f"    Writing DYNAMIC terrain scaling ({sector_scaling:.8f}) to offset {scaling_offset}")
                            write_float32(f, scaling_offset, sector_scaling)

                            print(f"    Writing {expected_data_size} bytes of TRANSFORMED image data from {tiff_filename} to offset {image_data_start_offset}")
                            f.seek(image_data_start_offset)
                            f.write(image_data_bytes)
                            # --- End Writes ---

                            print(f"    Successfully updated data and dynamic params for sector {sector_x:02d}_{sector_y:02d}.")
                            updated_sectors_count += 1
                            file_updated = True

                            # Advance offset past the written data
                            current_offset = image_data_start_offset + expected_data_size

                        except (IndexError, struct.error) as e:
                            print(f"  Error reading/processing metadata or data near match offset {match_offset} (Block start: {block_start_offset}): {e}")
                            current_offset = match_offset + 1
                            found_match = False
                            break
                        except Exception as e:
                             print(f"  Unexpected error processing block near match offset {match_offset}: {e}")
                             current_offset = match_offset + 1
                             found_match = False
                             break

                    # If a match was found, break inner pattern loop (offset already advanced)
                    if found_match:
                        break

                # If no pattern matched at current_offset, move to the next byte
                if not found_match:
                    current_offset += 1

    except FileNotFoundError:
        print(f"  Error: Output file disappeared or couldn't be opened: {output_heightmap_path}")
        return 0
    except Exception as e:
        print(f"  Fatal error processing file {output_heightmap_path}: {e}")
        # if file_updated: try: os.remove(output_heightmap_path); ... except OSError: pass
        return 0

    if updated_sectors_count == 0:
         print(f"  No sectors within the range [{MIN_SECTOR_INDEX}-{MAX_SECTOR_INDEX}] found or matched TIFs in {subfolder_name}.")
    else:
        print(f"  Finished processing {subfolder_name}. Updated data and dynamic params for {updated_sectors_count} sectors.")
    return updated_sectors_count


def main():
    print("Starting heightmap update process (DYNAMIC Params & Transformed Data)...")
    print("Calculates start height/scaling PER SECTOR based on TIFF min/max.")
    print("Transforms TIFF float heights to raw uint16 (0-65535) based on dynamic range.")
    print(f"Source Edited Images: '{EDITED_SECTORS_BASE_DIR}' (Assumes TIFFs contain float heights)")
    print(f"Original Data Location: '{ORIGINAL_DATA_BASE_DIR}'")
    print(f"Output Location: '{OUTPUT_BASE_DIR}'")
    print(f"Updating sectors in range: X={MIN_SECTOR_INDEX}-{MAX_SECTOR_INDEX}, Y={MIN_SECTOR_INDEX}-{MAX_SECTOR_INDEX}")
    print("-" * 30)

    total_updated_sectors = 0
    processed_folders = 0

    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create base output directory '{OUTPUT_BASE_DIR}': {e}")
        return

    try:
        if not os.path.isdir(EDITED_SECTORS_BASE_DIR):
             print(f"Error: Source directory '{EDITED_SECTORS_BASE_DIR}' not found.")
             return
        subfolders = [d for d in os.listdir(EDITED_SECTORS_BASE_DIR)
                      if os.path.isdir(os.path.join(EDITED_SECTORS_BASE_DIR, d))]
    except Exception as e:
        print(f"Error listing subdirectories in '{EDITED_SECTORS_BASE_DIR}': {e}")
        return

    if not subfolders:
        print(f"No subfolders (e.g., '000_000') containing TIFs found in '{EDITED_SECTORS_BASE_DIR}'. Nothing to process.")
        return

    print(f"Found {len(subfolders)} potential sector folders to process in '{EDITED_SECTORS_BASE_DIR}'.")

    for subfolder_name in sorted(subfolders):
        if len(subfolder_name) == 7 and subfolder_name[3] == '_' and subfolder_name[:3].isdigit() and subfolder_name[4:].isdigit():
             updated_count = update_heightmap_file_dynamic(subfolder_name) # Use the correct function name
             total_updated_sectors += updated_count
             if updated_count > 0 or os.path.exists(os.path.join(ORIGINAL_DATA_BASE_DIR, subfolder_name)):
                 processed_folders += 1
        else:
             print(f"Skipping '{subfolder_name}' in '{EDITED_SECTORS_BASE_DIR}' - does not match '###_###' format.")

    print("-" * 30)
    print("Processing complete.")
    print(f"Attempted processing for {processed_folders} heightmap files.")
    print(f"Total sectors updated (dynamic params and transformed data): {total_updated_sectors}")

if __name__ == "__main__":
    main()