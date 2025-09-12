import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


class HammingNetworkTrafficSigns:
    def __init__(self, img_size=100, num_patterns=3):
        """
        Red de Hamming para 3 clases: Pare, Ceda, Resalto.
        :param img_size: Tamaño de la imagen (100x100).
        :param num_patterns: Número de clases (3).
        """
        self.img_size = img_size
        self.input_size = (
            img_size * img_size
        )  # Vector por defecto: 100x100 = 10,000 bits
        self.weights = []  # Patrones (uno por clase)
        self.labels = ["pare", "ceda", "resalto"]  # Clases fijas

    def _preprocess_image(self, image_path):
        """
        Carga y binariza una imagen en escala de grises.
        :param image_path: Ruta a la imagen.
        :return: Vector binario aplanado (-1,1).
        """
        image = Image.open(image_path).convert("L")  # Escala de grises
        if image.size != (self.img_size, self.img_size):
            raise ValueError(
                f"La imagen debe ser {self.img_size}x{self.img_size} píxeles."
            )

        gray = np.array(image)  # Matriz [100, 100]
        binary = np.where(gray > 128, 1, 0)  # Binariza con umbral
        flat = binary.flatten()  # Vector de 10,000 elementos
        return np.where(flat == 0, -1, 1)  # A -1,1 para Hamming

    def _preprocess_image_rgb(self, image_path):
        """
        Carga y binariza una imagen RGB (1 bit por canal).
        :param image_path: Ruta a la imagen.
        :return: Vector binario aplanado (-1,1) con 3 bits por píxel.
        """
        image = Image.open(image_path).convert("RGB")
        if image.size != (self.img_size, self.img_size):
            raise ValueError(
                f"La imagen debe ser {self.img_size}x{self.img_size} píxeles."
            )

        rgb = np.array(image)  # Matriz [100, 100, 3]
        binary = np.where(rgb > 128, 1, 0)  # Binariza R,G,B
        flat = binary.flatten()  # Vector de 100x100x3 = 30,000 elementos
        return np.where(flat == 0, -1, 1)

    def train(self, dataset_dir, num_samples_per_class=3, use_rgb=False):
        """
        Entrena con imágenes de las clases pare, ceda, resalto.
        :param dataset_dir: Directorio del dataset.
        :param num_samples_per_class: Imágenes por clase para promediar.
        :param use_rgb: Si True, usa RGB (3 bits/píxel); si False, escala de grises.
        """
        preprocess_fn = (
            self._preprocess_image_rgb if use_rgb else self._preprocess_image
        )
        self.input_size = self.img_size * self.img_size * (3 if use_rgb else 1)

        for class_name in self.labels:
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Error: La carpeta {class_dir} no existe.")
                continue

            images = [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".ppm"))
            ]
            if not images:
                print(f"Error: No se encontraron imágenes en {class_name}.")
                continue
            if len(images) < num_samples_per_class:
                print(
                    f"Advertencia: Clase {class_name} tiene {len(images)} imágenes, usando todas."
                )
                num_samples_per_class = len(images)

            # Promedia imágenes por clase
            class_patterns = []
            for i in range(min(num_samples_per_class, len(images))):
                img_path = os.path.join(class_dir, images[i])
                pattern = preprocess_fn(img_path)
                class_patterns.append(pattern)

            # Calcula el patrón promedio
            avg_pattern = np.mean(class_patterns, axis=0)
            avg_pattern = np.where(avg_pattern > 0, 1, -1)  # Redondea a -1 o 1
            self.weights.append(avg_pattern)

        if len(self.weights) != 3:
            raise ValueError(
                f"Se espeeraban 3 clases, pero se entrenaron {len(self.weights)}."
            )
        print("Entrenamiento completado para las clases: pare, ceda, resalto.")

    def classify(self, image_path, use_rgb=False):
        """
        Clasifica una imagen de entrada.
        :param image_path: Ruta a la imagen de prueba.
        :param use_rgb: Si True, usa RGB; si False, escala de grises.
        :return: Etiqueta, distancia de Hamming, índice.
        """
        preprocess_fn = (
            self._preprocess_image_rgb if use_rgb else self._preprocess_image
        )
        input_pattern = preprocess_fn(image_path)

        min_distance = float("inf")
        best_label = None
        best_index = -1

        for i, weight in enumerate(self.weights):
            hamming_dist = np.sum(input_pattern != weight) / 2  # Distancia normalizada
            if hamming_dist < min_distance:
                min_distance = hamming_dist
                best_label = self.labels[i]
                best_index = i

        return best_label, min_distance, best_index

    def visualize(self, image_path, predicted_label, best_index, use_rgb=False):
        """
        Visualiza la imagen de entrada y el patrón predicho.
        """
        if use_rgb:
            input_img = np.array(Image.open(image_path).convert("RGB"))
            pred_pattern = np.where(self.weights[best_index] == 1, 255, 0).reshape(
                self.img_size, self.img_size, 3
            )
        else:
            input_img = np.array(Image.open(image_path).convert("L"))
            pred_pattern = np.where(self.weights[best_index] == 1, 255, 0).reshape(
                self.img_size, self.img_size
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(input_img, cmap="gray" if not use_rgb else None)
        ax1.set_title("Imagen de entrada")
        ax2.imshow(pred_pattern, cmap="gray" if not use_rgb else None)
        ax2.set_title(f"Predicción: {predicted_label}\nDistancia: {min_distance:.2f}")
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    dataset_dir = "dataset/"  # Reemplaza con la ruta a tu dataset
    test_image_path = (
        "dataset/test/test_0001.jpg"  # Reemplaza con tu imagen de prueba
    )
    img_size = 100
    use_rgb = True  # Cambia a True para usar RGB (más lento, más preciso)

    # Crea la red
    net = HammingNetworkTrafficSigns(img_size=img_size, num_patterns=3)

    # Entrena con 3 imágenes por clase
    net.train(dataset_dir, num_samples_per_class=3, use_rgb=use_rgb)

    # Clasifica una imagen de prueba
    label, min_distance, index = net.classify(test_image_path, use_rgb=use_rgb)

    print(f"Señal detectada: {label} (Distancia de Hamming: {min_distance:.2f})")

    # Visualiza
    net.visualize(test_image_path, label, index, use_rgb=use_rgb)
