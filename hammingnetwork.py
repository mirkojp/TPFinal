import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import csv


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
        self.output_dir = "dataset/test"  # Directorio para CSVs
        os.makedirs(self.output_dir, exist_ok=True)  # Crear directorio si no existe
        self.csv_input_path = os.path.join(
            self.output_dir, "datainput.csv"
        )  # Archivo CSV de entrada en dataset/test
        self.csv_output_path = os.path.join(
            self.output_dir, "dataoutput.csv"
        )  # Archivo CSV de salida en dataset/test

    def _preprocess_image(self, image_path):
        """
        Carga y binariza una imagen en escala de grises.
        """
        image = Image.open(image_path).convert("L")
        if image.size != (self.img_size, self.img_size):
            raise ValueError(
                f"La imagen debe ser {self.img_size}x{self.img_size} píxeles."
            )

        gray = np.array(image)
        binary = np.where(gray > 128, 1, 0)
        flat = binary.flatten()
        return np.where(flat == 0, -1, 1)

    def _preprocess_image_rgb(self, image_path, is_test=False):
        """
        Carga y binariza una imagen RGB (1 bit por canal).
        Si is_test=True, guarda el vector binarizado en el archivo CSV único.
        """
        image = Image.open(image_path).convert("RGB")
        if image.size != (self.img_size, self.img_size):
            raise ValueError(
                f"La imagen debe ser {self.img_size}x{self.img_size} píxeles."
            )

        rgb = np.array(image)
        binary = np.where(rgb > 128, 1, 0)
        flat = binary.flatten()
        binarized = np.where(flat == 0, -1, 1)

        return binarized

    def train(self, dataset_dir, num_samples_per_class=40, use_rgb=False):
        """
        Entrena con imágenes de las clases pare, ceda, resalto.
        """
        preprocess_fn = (
            self._preprocess_image_rgb if use_rgb else self._preprocess_image
        )
        self.input_size = self.img_size * self.img_size * (3 if use_rgb else 1)

        print("Iniciando entrenamiento...")
        print(f"Clases esperadas: {self.labels}")

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

            print(
                f"Procesando clase: {class_name} ({len(images)} imágenes disponibles)"
            )
            class_patterns = []
            for i in range(min(num_samples_per_class, len(images))):
                img_path = os.path.join(class_dir, images[i])
                print(f"  - Cargando imagen: {img_path}")
                pattern = preprocess_fn(img_path)  # No se pasa is_test=True
                class_patterns.append(pattern)

            # Promedia imágenes
            avg_pattern = np.mean(class_patterns, axis=0)
            avg_pattern = np.where(avg_pattern > 0, 1, -1)
            self.weights.append(avg_pattern)
            print(f"  - Patrón promedio para {class_name} almacenado.")

        if len(self.weights) != 3:
            raise ValueError(
                f"Se esperaban 3 clases, pero se entrenaron {len(self.weights)}."
            )
        print("Entrenamiento completado.")

    def classify(self, image_path, use_rgb=False):
        """
        Clasifica una imagen de entrada.
        """
        preprocess_fn = (
            self._preprocess_image_rgb if use_rgb else self._preprocess_image
        )
        input_pattern = preprocess_fn(image_path)

        print(f"Clasificando imagen: {image_path}")
        min_distance = float("inf")
        best_label = "none"
        best_index = -1

        for i, weight in enumerate(self.weights):
            hamming_dist = np.sum(input_pattern != weight) / 2
            print(f"  - Distancia a clase {self.labels[i]}: {hamming_dist:.2f}")
            if hamming_dist < min_distance:
                min_distance = hamming_dist
                best_label = self.labels[i]
                best_index = i

        # Always consider a pattern detected (Hamming network always chooses closest pattern)
        pattern_detected = 1
        detected_label = best_label

        return detected_label, min_distance, best_index, pattern_detected

    def inspect_network(self):
        """
        Muestra las clases y variables de la red.
        """
        print("\n=== Inspección de la red ===")
        print(f"Tamaño de imagen: {self.img_size}x{self.img_size}")
        print(f"Tamaño del vector de entrada: {self.input_size} bits")
        print(f"Clases definidas: {self.labels}")
        print(f"Número de patrones almacenados: {len(self.weights)}")
        if self.weights:
            print("Detalles de los patrones:")
            for i, (label, weight) in enumerate(zip(self.labels, self.weights)):
                print(f"  - Clase {label}: Patrón de {weight.shape[0]} bits")
                print(f"    Ejemplo de primeros 10 bits: {weight[:10]}")
        else:
            print("  - No hay patrones almacenados (entrenamiento no realizado).")

    def visualize_patterns(self, use_rgb=False):
        """
        Visualiza los patrones almacenados para cada clase.
        """
        if not self.weights:
            print("Error: No hay patrones almacenados para visualizar.")
            return

        fig, axes = plt.subplots(
            1, len(self.weights), figsize=(4 * len(self.weights), 4)
        )
        if len(self.weights) == 1:
            axes = [axes]  # Para una sola clase

        for i, (weight, label) in enumerate(zip(self.weights, self.labels)):
            if use_rgb:
                pattern = np.where(weight == 1, 255, 0).reshape(
                    self.img_size, self.img_size, 3
                )
            else:
                pattern = np.where(weight == 1, 255, 0).reshape(
                    self.img_size, self.img_size
                )

            axes[i].imshow(pattern, cmap="gray" if not use_rgb else None)
            axes[i].set_title(f"Patrón: {label}")
            axes[i].axis("off")
        plt.show()

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
        ax2.set_title(f"Predicción: {predicted_label}")
        plt.show()

    def process_test_images(self, test_dir, use_rgb=False):
        """
        Procesa las imágenes de prueba, carga datos desde datainput.csv y genera dataoutput.csv.
        """
        if not os.path.exists(self.csv_input_path):
            print(f"Error: El archivo {self.csv_input_path} no existe.")
            return

        # Leer datainput.csv
        input_data = []
        try:
            with open(self.csv_input_path, mode="r") as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) < 3:
                        print(f"Advertencia: Fila inválida en datainput.csv: {row}")
                        continue
                    image_name, has_pattern, pattern = row[0], row[1], row[2]
                    input_data.append(
                        {
                            "image_name": image_name,
                            "has_pattern": int(has_pattern),
                            "pattern": pattern,
                        }
                    )
            print(f"Se leyeron {len(input_data)} entradas desde {self.csv_input_path}")
        except Exception as e:
            print(f"Error al leer {self.csv_input_path}: {e}")
            return

        # Procesar imágenes y generar resultados
        output_data = []
        for entry in input_data:
            image_name = entry["image_name"]
            image_path = os.path.join(test_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Error: La imagen {image_path} no existe.")
                continue

            print(f"\nProcesando imagen de prueba: {image_path}")
            label, min_distance, index, pattern_detected = self.classify(
                image_path, use_rgb=use_rgb
            )
            print(
                f"Señal detectada: {label} (Distancia de Hamming: {min_distance:.2f})"
            )

            # Guardar resultados
            output_data.append(
                [
                    image_name,
                    entry["has_pattern"],
                    entry["pattern"],
                    pattern_detected,
                    label,
                ]
            )

            # Visualizar
            self.visualize(image_path, label, index, use_rgb=use_rgb)

        # Guardar resultados en dataoutput.csv
        try:
            with open(self.csv_output_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                # Escribir encabezado
                writer.writerow(
                    [
                        "image_name",
                        "has_pattern",
                        "pattern",
                        "pattern_detected",
                        "detected_pattern",
                    ]
                )
                # Escribir datos
                writer.writerows(output_data)
            print(f"Resultados guardados en: {self.csv_output_path}")
        except Exception as e:
            print(f"Error al guardar {self.csv_output_path}: {e}")


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    dataset_dir = "dataset/"  # Ruta al dataset
    test_dir = os.path.join(dataset_dir, "test")  # Directorio de imágenes de prueba
    img_size = 100
    use_rgb = True  # Usar RGB para procesar imágenes

    # Crea la red
    net = HammingNetworkTrafficSigns(img_size=img_size, num_patterns=3)

    # Inspecciona antes del entrenamiento
    print("Antes del entrenamiento:")
    net.inspect_network()

    # Entrena
    net.train(dataset_dir, num_samples_per_class=25, use_rgb=use_rgb)

    # Inspecciona después del entrenamiento
    print("\nDespués del entrenamiento:")
    net.inspect_network()

    # Visualiza los patrones almacenados
    print("\nVisualizando patrones almacenados:")
    net.visualize_patterns(use_rgb=use_rgb)

    # Procesa imágenes de prueba y genera dataoutput.csv
    net.process_test_images(test_dir, use_rgb=use_rgb)
