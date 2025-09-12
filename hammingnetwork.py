import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class HammingNetworkTrafficSigns:
    def __init__(self, img_size=8, num_patterns=10):  # img_size: lado de la imagen cuadrada (ej. 8x8)
        """
        Red de Hamming para señales de tránsito.
        :param img_size: Tamaño de la imagen redimensionada (cuadrada).
        :param num_patterns: Número máximo de patrones por clase (o total si no hay clases).
        """
        self.img_size = img_size
        self.input_size = img_size * img_size  # Vector aplanado (binario)
        self.weights = []  # Lista de patrones (uno por señal)
        self.labels = []   # Etiquetas de las señales (ej. "stop", "yield")
    
    def _preprocess_image(self, image_path):
        """
        Carga, redimensiona y binariza una imagen (escala de grises).
        :param image_path: Ruta a la imagen.
        :return: Vector binario aplanado (-1,1).
        """
        image = Image.open(image_path).convert('L')  # Convierte a escala de grises
        image = image.resize((self.img_size, self.img_size))  # Redimensiona
        gray = np.array(image)  # Matriz [img_size, img_size] con valores 0-255
        binary = np.where(gray > 128, 1, 0)  # Binariza con umbral
        flat = binary.flatten()
        return np.where(flat == 0, -1, 1)  # A -1,1 para Hamming
    
    def train(self, dataset_dir, num_samples_per_class=5):
        """
        Entrena con imágenes de un directorio (subdirectorios por clase).
        Ejemplo estructura: dataset/stop/image1.png, dataset/yield/image2.png
        :param dataset_dir: Directorio del dataset.
        :param num_samples_per_class: Muestras por clase para promediar patrones.
        """
        classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        self.labels = classes
        
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, class_name)
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(images) < num_samples_per_class:
                print(f"Advertencia: Clase {class_name} tiene pocas imágenes.")
                num_samples_per_class = len(images)
            
            # Promedia num_samples_per_class imágenes para un patrón robusto por clase
            class_patterns = []
            for i in range(num_samples_per_class):
                img_path = os.path.join(class_dir, images[i])
                pattern = self._preprocess_image(img_path)
                class_patterns.append(pattern)
            
            # Promedia los patrones (para reducir ruido)
            avg_pattern = np.mean(class_patterns, axis=0)
            avg_pattern = np.where(avg_pattern > 0, 1, -1)  # Redondea a -1 o 1
            self.weights.append(avg_pattern)
        
        print(f"Entrenado con {len(classes)} clases.")
    
    def classify(self, image_path):
        """
        Clasifica una imagen de entrada.
        :param image_path: Ruta a la imagen de prueba.
        :return: Etiqueta de la clase más cercana y distancia de Hamming.
        """
        input_pattern = self._preprocess_image(image_path)
        
        min_distance = float('inf')
        best_label = None
        best_index = -1
        
        for i, weight in enumerate(self.weights):
            # Distancia de Hamming: contar diferencias (o usar producto punto para similitud)
            hamming_dist = np.sum(input_pattern != weight) / 2  # Normalizada
            if hamming_dist < min_distance:
                min_distance = hamming_dist
                best_label = self.labels[i]
                best_index = i
        
        return best_label, min_distance, best_index
    
    def visualize(self, image_path, predicted_label, best_index):
        """
        Visualiza la imagen de entrada y el patrón predicho.
        """
        input_img = Image.open(image_path).convert('L').resize((self.img_size, self.img_size))
        input_array = np.array(input_img)
        
        pred_pattern = np.where(self.weights[best_index] == 1, 255, 0).reshape(self.img_size, self.img_size)
        
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(input_array, cmap='gray')
        axs[0].set_title('Entrada')
        axs[1].imshow(pred_pattern, cmap='gray')
        axs[1].set_title(f'Patrón: {predicted_label}')
        axs[2].text(0.5, 0.5, f'Distancia Hamming: {predicted_label}', ha='center', va='center', transform=axs[2].transAxes)
        axs[2].axis('off')
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Asume un dataset simple en 'dataset/' con subdirs como 'stop', 'yield', etc.
    # Descarga imágenes de señales y organízalas en subdirectorios por clase.
    dataset_dir = "dataset/"  # Reemplaza con tu directorio
    
    # Crea la red (imágenes 8x8 para simplicidad)
    net = HammingNetworkTrafficSigns(img_size=8, num_patterns=10)
    
    # Entrena (promedia 3 imágenes por clase)
    net.train(dataset_dir, num_samples_per_class=3)
    
    # Clasifica una imagen de prueba
    test_image = "test_stop.png"  # Ruta a una imagen de prueba
    label, distance, index = net.classify(test_image)
    
    print(f"Señal detectada: {label} (Distancia de Hamming: {distance:.2f})")
    
    # Visualiza
    net.visualize(test_image, label, index)