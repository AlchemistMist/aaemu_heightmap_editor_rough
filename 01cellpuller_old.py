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
    # Ensure coordinates are numeric before calculation
    try:
        x = math.floor((float(startx) - float(firstStartX)) / 64)
        y = math.floor((float(starty) - float(firstStartY)) / 64)
        return x, y
    except (ValueError, TypeError) as e:
        print(f"Error calculating sector for ({startx}, {starty}) with base ({firstStartX}, {firstStartY}): {e}")
        return None, None # Indicate error


def collect_heightmap_data(file_data, folder_name, first_start_x, first_start_y):
    """
    Collects heightmap metadata and raw image data from the binary file.
    Corrected logic based on search string being map_string + var_string.
    """
    results = []
    data_length = len(file_data)
    search_string_length = 8 # All search strings are 8 bytes

    # Iterate through the file data looking for the search strings
    # Need at least 37 bytes before the search string and 8 bytes for it
    for i in range(data_length - search_string_length + 1):
        current_slice = file_data[i : i + search_string_length]

        if current_slice in SEARCH_STRINGS:
            # If a match is found at index 'i', this 'i' marks the START of map_string
            # The full 45-byte metadata block starts 37 bytes *before* this index 'i'
            block_start_offset = i - 37

            # Check if the calculated start offset is valid (not negative)
            if block_start_offset < 0:
                continue # Cannot have metadata starting before the file begins

            try:
                entry = {}

                # --- Read map_string and var_string DIRECTLY from the found position 'i' ---
                entry['map_string'] = read_uint32(file_data, i) # Read resolution from i to i+3
                entry['var_string'] = read_uint32(file_data, i + 4) # Read var_string from i+4 to i+7

                # --- Read the PRECEDING metadata fields relative to block_start_offset ---
                entry['offset'] = block_start_offset # Store the actual start of the block
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
                # map_string and var_string are already read

                # --- Logic from Lua to adjust startX/startY if not whole numbers ---
                # Using the more robust isclose for floating point comparisons
                if not math.isclose(entry['start_x'] % 1, 0.0, abs_tol=1e-9) and not math.isclose(entry['start_x'] % 1, 1.0, abs_tol=1e-9):
                    # print(f"Adjusting start_x ({entry['start_x']}) for folder {folder_name}")
                    entry['start_x'] = first_start_x + 1088 # Value from Lua script
                if not math.isclose(entry['start_y'] % 1, 0.0, abs_tol=1e-9) and not math.isclose(entry['start_y'] % 1, 1.0, abs_tol=1e-9):
                    # print(f"Adjusting start_y ({entry['start_y']}) for folder {folder_name}")
                    entry['start_y'] = first_start_y + 1088 # Value from Lua script
                # --- End adjustment logic ---

                # Calculate sector names
                sector_x, sector_y = check_sector(entry['start_x'], entry['start_y'], first_start_x, first_start_y)
                if sector_x is None: # Skip if sector calculation failed
                    print(f"Skipping entry near block offset {block_start_offset} due to sector calculation error.")
                    continue
                entry['sector_name_x'] = sector_x
                entry['sector_name_y'] = sector_y
                entry['folder_name'] = folder_name

                # --- Extract Image Data ---
                resolution = entry['map_string']
                if resolution <= 0:
                     print(f"Warning: Invalid resolution ({resolution}) found starting at map_string offset {i}. Skipping image data extraction.")
                     continue # Cannot process image data with invalid resolution

                # Image data starts immediately AFTER var_string (which ends at i + 7)
                image_data_start_offset = i + 8
                bytes_per_pixel = 2 # 16-bit grayscale
                image_data_size = resolution * resolution * bytes_per_pixel

                if image_data_start_offset + image_data_size > data_length:
                    print(f"Warning: Not enough data for image at map_string offset {i}. Expected {image_data_size} bytes starting at {image_data_start_offset}, file size {data_length}.")
                    continue # Skip if data is incomplete

                entry['image_data_bytes'] = file_data[image_data_start_offset : image_data_start_offset + image_data_size]

                results.append(entry)

            except (IndexError, struct.error) as e:
                print(f"Error reading metadata or image data near block offset {block_start_offset} (map_string offset {i}): {e}")
                # Continue searching even if one entry fails
                continue
            except Exception as e:
                print(f"Unexpected error processing entry near block offset {block_start_offset} (map_string offset {i}): {e}")
                continue

    return results

def save_image_data(entry):
    """
    Saves the extracted image data as a 16-bit grayscale TIFF after
    applying the transformation formula:
    new = round(clip(((scaling * old) + min_height) / 2048 * 65535, 0, 65535))
    """
    folder_name = entry.get('folder_name', 'unknown_folder')
    sector_x = entry.get('sector_name_x', 'X')
    sector_y = entry.get('sector_name_y', 'Y')
    resolution = entry.get('map_string')
    image_bytes = entry.get('image_data_bytes')
    scaling_value = entry.get('scaling_value')
    min_height = entry.get('min_height')

    # --- Input Validation ---
    if not image_bytes:
        print(f"Skipping save for sector {sector_x},{sector_y} in {folder_name}: Missing image data.")
        return
    if resolution is None or resolution <= 0:
        print(f"Skipping save for sector {sector_x},{sector_y} in {folder_name}: Invalid resolution ({resolution}).")
        return
    if scaling_value is None:
         print(f"Skipping save for sector {sector_x},{sector_y} in {folder_name}: Missing 'scaling_value'.")
         return
    if min_height is None:
         print(f"Skipping save for sector {sector_x},{sector_y} in {folder_name}: Missing 'min_height'.")
         return

    # Create output directory structure
    output_dir_base = "output_sectors"
    output_dir_folder = os.path.join(output_dir_base, folder_name)
    os.makedirs(output_dir_folder, exist_ok=True)

    # Construct filename
    output_filename = f"sector_{sector_x:02d}_{sector_y:02d}.tif"
    output_filepath = os.path.join(output_dir_folder, output_filename)

    try:
        # --- Data Loading ---
        # Convert raw bytes to numpy array (16-bit unsigned integer)
        expected_size = resolution * resolution * 2
        if len(image_bytes) != expected_size:
             print(f"Error: Image data size mismatch for {output_filepath}. Expected {expected_size}, got {len(image_bytes)}. Skipping save.")
             return

        # Load raw uint16 data
        image_array_1d_uint16 = np.frombuffer(image_bytes, dtype=np.uint16)

        # --- Transformation ---
        # 1. Convert to float for accurate calculations
        image_array_1d_float = image_array_1d_uint16.astype(np.float64) # Use float64 for better precision

        # 2. Define constants for the formula
        new_height_divisor = 2048.0  # Use float for division
        max_uint16_value = 65535.0 # Use float for multiplication/scaling target

        # 3. Apply the formula: new = ((scaling * old) + min_height) / new_height * 65535
        # NumPy handles element-wise operations automatically
        transformed_float = (
            (scaling_value * image_array_1d_float + min_height) /
             new_height_divisor * max_uint16_value
        )

        # 4. Round to the nearest whole number
        rounded_float = np.round(transformed_float)

        # 5. Clip values to ensure they are within the uint16 range [0, 65535]
        clipped_float = np.clip(rounded_float, 0, 65535)

        # 6. Convert back to uint16
        final_image_array_1d_uint16 = clipped_float.astype(np.uint16)

        # --- Reshape and Save ---
        # Reshape the *transformed* 1D array into 2D array (height, width)
        image_array_2d_transformed = final_image_array_1d_uint16.reshape((resolution, resolution))

        # Save the transformed data as 16-bit grayscale TIFF
        tifffile.imwrite(output_filepath, image_array_2d_transformed)
        # print(f"Saved transformed: {output_filepath}")

    except ValueError as e:
        # Error during reshape likely means data size issue calculated earlier or formula error
        print(f"Error processing image data for {output_filepath}: {e}. Check data consistency and formula inputs.")
        print(f"  Input array size: {image_array_1d_uint16.size}, Target shape: ({resolution}, {resolution})")
    except Exception as e:
        print(f"Failed to save transformed TIFF file {output_filepath}: {e}")


# --- Main Processing Logic ---
def main():
    base_input_path = "." # Assume the script is run where 000_000, 001_000 etc. folders are
    output_dir_base = "output_sectors_grey"

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
        print(f"Created base output directory: {output_dir_base}")

    # Iterate through potential folder names (adjust range if needed)
    # Ranges from the Lua script
    for row in range(36): # 0 to 35
        for col in range(39): # 0 to 38
            folder_name = f"{row:03d}_{col:03d}"
            # Construct the path to the specific heightmap.dat file
            heightmap_path = os.path.join(base_input_path, folder_name, "client", "terrain", "heightmap.dat")

            if os.path.exists(heightmap_path):
                print(f"Processing file: {heightmap_path}")
                try:
                    with open(heightmap_path, "rb") as f:
                        file_data = f.read()

                    if len(file_data) < 172: # Need at least 172 bytes to read firstStartX/Y
                         print(f"Warning: File {heightmap_path} is too small ({len(file_data)} bytes). Skipping.")
                         continue

                    # Read first StartX and StartY needed for sector calculation
                    # Offsets are from the Lua script (164, 168)
                    first_start_x = read_float32(file_data, 164)
                    first_start_y = read_float32(file_data, 168)

                    # Collect metadata and image data
                    heightmap_entries = collect_heightmap_data(file_data, folder_name, first_start_x, first_start_y)

                    if not heightmap_entries:
                        print(f"No heightmap sectors found in {heightmap_path}")
                        continue

                    # Save each found sector as a TIFF image
                    for entry in heightmap_entries:
                         if 'image_data_bytes' in entry:
                             save_image_data(entry)
                         else:
                            print(f"Skipping save for sector {entry.get('sector_name_x','N/A')},{entry.get('sector_name_y','N/A')} in {folder_name} - missing image data.")


                    # --- Optional: Check for missing sectors (like the Lua script) ---
                    # actual_sectors = {f"{e['sector_name_x']},{e['sector_name_y']}" for e in heightmap_entries}
                    # expected_sectors = {f"{r},{c}" for r in range(16) for c in range(16)}
                    # missing_sectors = expected_sectors - actual_sectors
                    # if missing_sectors:
                    #     print(f"Missing sectors in {folder_name}: {sorted(list(missing_sectors))}")
                    # --- End optional check ---

                except FileNotFoundError:
                    # This case is already handled by os.path.exists, but good practice
                    print(f"File not found during processing: {heightmap_path}")
                except PermissionError:
                     print(f"Permission denied reading file: {heightmap_path}")
                except (IndexError, struct.error) as e:
                     print(f"Error reading initial data (first StartX/Y?) from {heightmap_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred processing {heightmap_path}: {e}")
            # else:
            #     # Optional: print message if a specific file doesn't exist
            #     # print(f"File does not exist, skipping: {heightmap_path}")
            #     pass


if __name__ == "__main__":
    main()
    print("\nProcessing finished.")