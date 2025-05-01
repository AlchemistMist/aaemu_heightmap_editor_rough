import os
from PIL import Image

# Define input and output folders
input_folder = "tiff_resize" 
output_folder = "raw_input"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
        # Construct full file path
        input_path = os.path.join(input_folder, filename)
        
        # Open the TIFF image
        with Image.open(input_path) as img:
            # Ensure it's a 16-bit grayscale image
            if img.mode != 'I;16':
                print(f"Skipping {filename}: Image is not 16-bit grayscale.")
                continue
            
            # Create a new blank canvas with the specified size (4096x4096)
            new_size = (4096, 4096)
            new_img = Image.new('I;16', new_size)

            # Paste the original image in the bottom left corner
            new_img.paste(img, (0, new_size[1] - img.size[1]))

            # Construct output file path
            output_path = os.path.join(output_folder, filename)

            # Save the new image
            new_img.save(output_path)
            print(f"Processed {filename}: Resized and saved.")

print("All done!")