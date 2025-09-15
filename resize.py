import cv2
import numpy as np
import os

def resize_image(image_path, output_path, target_size=(100, 100)):
    """
    Redimensiona una imagen manteniendo la relación de aspecto y la ajusta a un tamaño objetivo.
    Si la imagen redimensionada es más pequeña que el tamaño objetivo, la centra en un lienzo negro.

    Args:
        image_path (str): Ruta de la imagen de entrada.
        output_path (str): Ruta donde se guardará la imagen procesada.
        target_size (tuple): Tamaño objetivo en píxeles (ancho, alto).
    """
    # Carga la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Obtiene las dimensiones originales
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calcula la relación de aspecto
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    # Redimensiona manteniendo la relación de aspecto
    if aspect_ratio > target_aspect:
        # Imagen más ancha que el objetivo, ajusta al ancho
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # Imagen más alta que el objetivo, ajusta al alto
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    # Redimensiona la imagen
    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA if max(h, w) > max(target_w, target_h) else cv2.INTER_CUBIC
        ),
    )

    # Si la imagen es más pequeña que el objetivo, la centra en un lienzo negro
    if new_w < target_w or new_h < target_h:
        # Crea un lienzo negro del tamaño objetivo
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # Calcula el padding para centrar la imagen
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        # Coloca la imagen redimensionada en el lienzo
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image
        final_image = canvas
    else:
        final_image = resized_image

    # Guarda la imagen procesada
    cv2.imwrite(output_path, final_image)
    print(f"Image saved to {output_path}")

# Procesa todas las imágenes de una carpeta
input_folder = "images/nuevas"
output_folder = "images/nuevas Resized"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"resized_{filename}")
        resize_image(input_path, output_path)