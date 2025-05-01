import numpy as np
import tifffile as tiff
import os
import sys
import math # Import math for square root

# --- Configuration ---
INPUT_FOLDER = "edited_raw"
OUTPUT_FOLDER = "edited_tiff"
# We don't need fixed width/height constants anymore
# --- End Configuration ---

def convert_raw_to_tiff(raw_file_path, tiff_file_path):
    """
    Reads a 16-bit grayscale SQUARE RAW image file, calculates its dimensions,
    and saves it as a 16-bit grayscale TIFF file.

    Args:
        raw_file_path (str): Path to the input RAW file.
        tiff_file_path (str): Path to the output TIFF file.

    Raises:
        ValueError: If the file size is invalid (e.g., odd number of bytes,
                    doesn't correspond to a perfect square number of pixels).
        FileNotFoundError: If the input file doesn't exist.
    """
    try:
        # --- Calculate Dimensions from File Size ---
        actual_size_bytes = os.path.getsize(raw_file_path)

        if actual_size_bytes == 0:
            print(f"Warning: Skipping empty file {os.path.basename(raw_file_path)}.")
            return # Skip empty files

        # Each pixel is 2 bytes (uint16)
        bytes_per_pixel = 2
        if actual_size_bytes % bytes_per_pixel != 0:
            raise ValueError(f"Invalid file size ({actual_size_bytes} bytes). "
                             f"Size must be an even number for 16-bit pixels.")

        total_pixels = actual_size_bytes // bytes_per_pixel

        # Calculate the side length (square root of total pixels)
        side_float = math.sqrt(total_pixels)

        # Check if the side length is an integer (i.e., a perfect square)
        if side_float % 1 != 0:
            raise ValueError(f"File size ({actual_size_bytes} bytes) results in "
                             f"{total_pixels} pixels, which is not a perfect square. "
                             f"Cannot determine dimensions for a square image.")

        # If it's an integer, cast it to int
        side_length = int(side_float)
        width = side_length
        height = side_length
        # --- End Dimension Calculation ---

        # Read the raw binary data as 16-bit unsigned integers (little-endian is common)
        # If your RAW data is big-endian, use dtype='>u2' instead of np.uint16
        image_data_1d = np.fromfile(raw_file_path, dtype=np.uint16)

        # Sanity check: ensure the number of elements read matches calculation
        if image_data_1d.size != total_pixels:
             # This should ideally not happen if size checks passed, but good practice
             raise ValueError(f"Read {image_data_1d.size} pixels, but calculated {total_pixels} "
                              f"based on file size. File might be corrupt or read incorrectly.")

        # Reshape the 1D array into a 2D image (height, width)
        image_data_2d = image_data_1d.reshape((height, width))

        # Write the 2D numpy array to a TIFF file
        # The dtype is already uint16, so tifffile will save it as 16-bit grayscale
        tiff.imwrite(tiff_file_path, image_data_2d)

        print(f"Successfully converted {os.path.basename(raw_file_path)} ({width}x{height}) "
              f"to {os.path.basename(tiff_file_path)}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {raw_file_path}")
        raise # Re-raise to be caught in the main loop for error counting
    except ValueError as ve:
        print(f"Error processing {os.path.basename(raw_file_path)}: {ve}")
        raise # Re-raise
    except Exception as e:
        print(f"An unexpected error occurred while converting {os.path.basename(raw_file_path)}: {e}")
        raise # Re-raise


if __name__ == "__main__":
    # Check if input folder exists
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        print("Please create the folder and place your .raw files inside.")
        sys.exit(1) # Exit the script if the input folder is missing

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # exist_ok=True prevents error if folder exists

    print(f"Starting RAW to TIFF conversion...")
    print(f"Input folder: '{INPUT_FOLDER}'")
    print(f"Output folder: '{OUTPUT_FOLDER}'")
    print(f"Assuming ALL RAW images are SQUARE and contain 16-bit unsigned integer data (uint16).")
    print(f"Dimensions will be calculated based on file size.")
    print("-" * 30)

    file_count = 0
    conversion_count = 0
    error_count = 0

    # Process all .raw files in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        # Check if the file ends with .raw (case-insensitive)
        if filename.lower().endswith('.raw'):
            file_count += 1
            raw_file_path = os.path.join(INPUT_FOLDER, filename)
            # Create the output filename by replacing .raw with .tif
            base_filename = os.path.splitext(filename)[0] # Get filename without extension
            tiff_filename = base_filename + ".tif" # Use .tif extension
            tiff_file_path = os.path.join(OUTPUT_FOLDER, tiff_filename)

            try:
                # Check if it's actually a file and not a directory
                if os.path.isfile(raw_file_path):
                    # Call the conversion function (no width/height needed as args)
                    convert_raw_to_tiff(raw_file_path, tiff_file_path)
                    conversion_count += 1
                else:
                     print(f"Skipping item (not a file): {filename}")
            except Exception:
                 # Error message is printed inside the function
                 error_count += 1

    print("-" * 30)
    print("Conversion process finished.")
    print("\n--- Conversion Summary ---")
    print(f"Total potential .raw files found: {file_count}")
    print(f"Successfully converted: {conversion_count}")
    print(f"Errors/Skipped files: {error_count}")
    print("-" * 30)