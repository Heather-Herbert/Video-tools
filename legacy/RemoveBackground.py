import os
from rembg import remove
from PIL import Image


def strip_background(input_path):
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Create the output filename
    file_name, _ = os.path.splitext(input_path)
    output_path = f"{file_name}_no_bg.png"

    try:
        print(f"Processing: {input_path}...")

        # Open the input image
        with open(input_path, 'rb') as i:
            input_data = i.read()

            # Remove the background
            output_data = remove(input_data)

            # Save the result
            with open(output_path, 'wb') as o:
                o.write(output_data)

        print(f"Success! Saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Usage
if __name__ == "__main__":
    # Replace with your image filename (e.g., 'photo.jpg' or 'image.png')
    target_image = "/home/heather/Documents/youTube notes/17-March-2026/untitled-f049098.png"
    strip_background(target_image)