import cv2
import numpy as np
import os


def resize_image(image_path, output_path, target_size=(100, 100)):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Get original dimensions
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate aspect ratio
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    # Resize while preserving aspect ratio
    if aspect_ratio > target_aspect:
        # Image is wider than target, fit to width
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # Image is taller than target, fit to height
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    # Resize image
    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA if max(h, w) > max(target_w, target_h) else cv2.INTER_CUBIC
        ),
    )

    # If image is smaller than 100x100, pad it with black pixels
    if new_w < target_w or new_h < target_h:
        # Create a black canvas of target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # Calculate padding
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        # Place resized image on canvas
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image
        final_image = canvas
    else:
        final_image = resized_image

    # Save the output image
    cv2.imwrite(output_path, final_image)
    print(f"Image saved to {output_path}")


# Example usage for a single image
# input_image = "Ceda0001.jpg"
# output_image = "resized_image.jpg"
# resize_image(input_image, output_image)

# Example usage for a folder of images
input_folder = "images/Resalto"
output_folder = "images/Resalto Resized"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"resized_{filename}")
        resize_image(input_path, output_path)
