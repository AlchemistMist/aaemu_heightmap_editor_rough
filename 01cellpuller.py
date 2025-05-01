import struct
import os
import math
import numpy as np
import tifffile

# Define the search patterns (converted to bytes)
# These patterns seem to identify the start of metadata blocks
SEARCH_STRINGS = [
    bytes([0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00]),
    bytes([0x09, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
    bytes([0x11, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
    bytes([0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
    bytes([0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
    bytes([0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
    bytes([0x11, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]),
    bytes([0x11, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x0B, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x0D, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00]),
    bytes([0x21, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00])
]

# --- Define the valid range for sector indices ---
MIN_SECTOR_INDEX = 0
MAX_SECTOR_INDEX = 15
# --- End Configuration ---


# Helper functions to read specific data types from binary data
def read_float32(data, offset):
    """Reads a 32-bit little-endian float from bytes."""
    if offset + 4 > len(data):
        raise IndexError("Offset out of bounds for reading float32")
    return struct.unpack('<f', data[offset:offset+4])[0]

def read_uint32(data, offset):
    """Reads a 32-bit little-endian unsigned integer from bytes."""
    if offset + 4 > len(data):
        raise IndexError("Offset out of bounds for reading uint32")
    return struct.unpack('<I', data[offset:offset+4])[0]

def read_byte(data, offset):
    """Reads a single byte from bytes."""
    if offset + 1 > len(data):
         raise IndexError("Offset out of bounds for reading byte")
    return struct.unpack('<B', data[offset:offset+1])[0]

def check_sector(startx, starty, firstStartX, firstStartY):
    """Calculate sectors based on given coordinates."""
    try:
        x = math.floor((float(startx) - float(firstStartX)) / 64)
        y = math.floor((float(starty) - float(firstStartY)) / 64)
        return x, y
    except (ValueError, TypeError) as e:
        print(f"Error calculating sector for ({startx}, {starty}) with base ({firstStartX}, {firstStartY}): {e}")
        return None, None # Indicate error


def collect_heightmap_data(file_data, folder_name, first_start_x, first_start_y):
    """
    Collects heightmap metadata and raw image data from the binary file,
    filtering to include only sectors within the 0-15 index range.
    """
    results = []
    data_length = len(file_data)
    search_string_length = 8 # All search strings are 8 bytes
    skipped_outside_range = 0

    # Iterate through the file data looking for the search strings
    for i in range(data_length - search_string_length + 1):
        current_slice = file_data[i : i + search_string_length]

        if current_slice in SEARCH_STRINGS:
            block_start_offset = i - 37
            if block_start_offset < 0:
                continue

            try:
                entry = {}
                entry['map_string'] = read_uint32(file_data, i)
                entry['var_string'] = read_uint32(file_data, i + 4)
                entry['offset'] = block_start_offset
                entry['start_string'] = read_uint32(file_data, block_start_offset + 0)
                entry['start_x'] = read_float32(file_data, block_start_offset + 4)
                entry['start_y'] = read_float32(file_data, block_start_offset + 8)
                entry['min_z'] = read_float32(file_data, block_start_offset + 12)
                entry['finish_x'] = read_float32(file_data, block_start_offset + 16)
                entry['finish_y'] = read_float32(file_data, block_start_offset + 20)
                entry['max_z'] = read_float32(file_data, block_start_offset + 24)
                entry['padding'] = read_byte(file_data, block_start_offset + 28)
                entry['min_height'] = read_float32(file_data, block_start_offset + 29)
                entry['scaling_value'] = read_float32(file_data, block_start_offset + 33)

                # Adjust startX/startY if needed
                if not math.isclose(entry['start_x'] % 1, 0.0, abs_tol=1e-9) and not math.isclose(entry['start_x'] % 1, 1.0, abs_tol=1e-9):
                    entry['start_x'] = first_start_x + 1088
                if not math.isclose(entry['start_y'] % 1, 0.0, abs_tol=1e-9) and not math.isclose(entry['start_y'] % 1, 1.0, abs_tol=1e-9):
                    entry['start_y'] = first_start_y + 1088

                # Calculate sector names
                sector_x, sector_y = check_sector(entry['start_x'], entry['start_y'], first_start_x, first_start_y)
                if sector_x is None:
                    print(f"Skipping entry near block offset {block_start_offset} due to sector calculation error.")
                    continue

                # --- FILTERING: Only include sectors within 0-15 range ---
                if not (MIN_SECTOR_INDEX <= sector_x <= MAX_SECTOR_INDEX and MIN_SECTOR_INDEX <= sector_y <= MAX_SECTOR_INDEX):
                    skipped_outside_range += 1
                    continue # Skip the rest of the processing for this entry
                # --- END FILTERING ---

                # If the filter passed, store the calculated sector names
                entry['sector_name_x'] = sector_x
                entry['sector_name_y'] = sector_y
                entry['folder_name'] = folder_name

                # --- Extract Image Data (only for valid sectors) ---
                resolution = entry['map_string']
                if resolution <= 0:
                     print(f"Warning: Invalid resolution ({resolution}) found for valid sector {sector_x},{sector_y} at offset {i}. Skipping.")
                     continue

                image_data_start_offset = i + 8
                bytes_per_pixel = 2 # 16-bit grayscale
                image_data_size = resolution * resolution * bytes_per_pixel

                if image_data_start_offset + image_data_size > data_length:
                    print(f"Warning: Not enough data for image of valid sector {sector_x},{sector_y} at offset {i}. Expected {image_data_size} bytes. Skipping.")
                    continue

                entry['image_data_bytes'] = file_data[image_data_start_offset : image_data_start_offset + image_data_size]

                # Append the complete entry only if it passed the filter and image data was extracted
                results.append(entry)

            except (IndexError, struct.error) as e:
                print(f"Error reading metadata or image data near block offset {block_start_offset} (map_string offset {i}): {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing entry near block offset {block_start_offset} (map_string offset {i}): {e}")
                continue

    if skipped_outside_range > 0:
         print(f"  Skipped {skipped_outside_range} sectors with coordinates outside the {MIN_SECTOR_INDEX}-{MAX_SECTOR_INDEX} range.")

    return results

# --- save_image_data function remains unchanged ---
def save_image_data(entry):
    """
    Saves the extracted image data as a 16-bit grayscale TIFF after
    applying the transformation formula and using zero-padded filenames.
    Filename format: sector_XX_YY.tif
    Formula: new = round(clip(((scaling * old) + min_height) / 2048 * 65535, 0, 65535))
    """
    folder_name = entry.get('folder_name', 'unknown_folder')
    # These sector coordinates are guaranteed to be within 0-15 now
    sector_x = entry.get('sector_name_x')
    sector_y = entry.get('sector_name_y')
    resolution = entry.get('map_string')
    image_bytes = entry.get('image_data_bytes')
    scaling_value = entry.get('scaling_value')
    min_height = entry.get('min_height')

    # Input Validation (should be less critical now but keep for robustness)
    if not image_bytes: return
    if sector_x is None or sector_y is None: return
    if resolution is None or resolution <= 0: return
    if scaling_value is None: return
    if min_height is None: return

    # Output directory structure
    output_dir_base = "output_sectors" # Suggest changing dir name
    output_dir_folder = os.path.join(output_dir_base, folder_name)
    os.makedirs(output_dir_folder, exist_ok=True)

    # Construct filename WITH zero-padding
    output_filename = f"sector_{sector_x:02d}_{sector_y:02d}.tif"
    output_filepath = os.path.join(output_dir_folder, output_filename)

    try:
        # Data Loading
        expected_size = resolution * resolution * 2
        if len(image_bytes) != expected_size:
             print(f"Error: Image data size mismatch for {output_filepath}. Expected {expected_size}, got {len(image_bytes)}. Skipping save.")
             return
        image_array_1d_uint16 = np.frombuffer(image_bytes, dtype=np.uint16)

        # Transformation
        image_array_1d_float = image_array_1d_uint16.astype(np.float64)
        new_height_divisor = 2048.0
        max_uint16_value = 65535.0
        transformed_float = (
            (scaling_value * image_array_1d_float + min_height) /
             new_height_divisor * max_uint16_value
        )
        rounded_float = np.round(transformed_float)
        clipped_float = np.clip(rounded_float, 0, 65535)
        final_image_array_1d_uint16 = clipped_float.astype(np.uint16)

        # Reshape and Save
        image_array_2d_transformed = final_image_array_1d_uint16.reshape((resolution, resolution))
        tifffile.imwrite(output_filepath, image_array_2d_transformed)

    except ValueError as e:
        print(f"Error processing image data for {output_filepath}: {e}.")
    except TypeError as e:
        print(f"Error formatting filename (likely non-integer sector coords) for {output_filepath}: {e}")
    except Exception as e:
        print(f"Failed to save transformed TIFF file {output_filepath}: {e}")


# --- Main Processing Logic remains unchanged ---
def main():
    base_input_path = "."
    output_dir_base = "output_sectors" # Use the potentially new name

    # --- Check if output dir exists or create it ---
    # It's better to check/create here once rather than in every save call potentially
    try:
        os.makedirs(output_dir_base, exist_ok=True)
        print(f"Ensured base output directory exists: {output_dir_base}")
    except OSError as e:
        print(f"Error: Could not create base output directory '{output_dir_base}': {e}")
        return # Cannot proceed if output dir cannot be made

    print(f"\nStarting processing. Only sectors within range {MIN_SECTOR_INDEX}-{MAX_SECTOR_INDEX} will be saved.")

    total_files_processed = 0
    total_sectors_saved = 0

    for row in range(36): # 0 to 35
        for col in range(39): # 0 to 38
            folder_name = f"{row:03d}_{col:03d}"
            heightmap_path = os.path.join(base_input_path, folder_name, "client", "terrain", "heightmap.dat")

            if os.path.exists(heightmap_path):
                print(f"Processing file: {heightmap_path}")
                total_files_processed += 1
                file_sectors_saved = 0
                try:
                    with open(heightmap_path, "rb") as f:
                        file_data = f.read()

                    if len(file_data) < 172:
                         print(f"  Warning: File {heightmap_path} is too small ({len(file_data)} bytes). Skipping.")
                         continue

                    first_start_x = read_float32(file_data, 164)
                    first_start_y = read_float32(file_data, 168)

                    # collect_heightmap_data now returns only filtered entries
                    heightmap_entries = collect_heightmap_data(file_data, folder_name, first_start_x, first_start_y)

                    if not heightmap_entries:
                        print(f"  No valid heightmap sectors (within range {MIN_SECTOR_INDEX}-{MAX_SECTOR_INDEX}) found in this file.")
                        continue

                    for entry in heightmap_entries:
                         if 'image_data_bytes' in entry:
                             save_image_data(entry)
                             file_sectors_saved += 1
                         # We shouldn't get entries without image_data_bytes if filtering works correctly,
                         # unless there was an error during extraction post-filtering.

                    if file_sectors_saved > 0:
                        print(f"  Saved {file_sectors_saved} sectors from this file.")
                        total_sectors_saved += file_sectors_saved

                except Exception as e:
                    # Catch potential errors during file processing
                    print(f"An error occurred processing {heightmap_path}: {e}")
                    # Decide if you want to continue to the next file or stop
                    continue

    print(f"\nProcessing finished.")
    print(f"Scanned folders potentially containing {total_files_processed} heightmap.dat files.")
    print(f"Total sectors saved (within range {MIN_SECTOR_INDEX}-{MAX_SECTOR_INDEX}): {total_sectors_saved}")


if __name__ == "__main__":
    main()