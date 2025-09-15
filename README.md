# TPFinal - Reconocimiento de Señales de Tránsito con Red de Hamming

Este proyecto implementa una red de Hamming para el reconocimiento automático de señales de tránsito (pare, ceda, resalto) a partir de imágenes. Incluye scripts para el procesamiento de imágenes, entrenamiento, clasificación y evaluación de resultados.

## Estructura del Proyecto

```
TPFinal/
├── dataset/
│   ├── pare/
│   ├── ceda/
│   ├── resalto/
│   └── test/
│       ├── datainput.csv
│       └── (imágenes de prueba)
├── hammingnetwork.py
├── resize.py
├── metrics.py
├── README.md
```

## Scripts Principales

- **hammingnetwork.py**  
  Implementa la red de Hamming, permite entrenar con imágenes, clasificar nuevas imágenes, visualizar patrones y procesar lotes de prueba.

- **resize.py**  
  Redimensiona imágenes manteniendo la relación de aspecto y las ajusta a un tamaño estándar, centrando en un lienzo negro si es necesario.

- **metrics.py**  
  Calcula métricas de desempeño (matriz de confusión, precisión, reporte de clasificación) a partir de los resultados generados por la red.

## Requisitos

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pillow
- opencv-python

Instala los requisitos con:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow opencv-python
```

## Uso


1. **Entrenar y probar la red:**  
   Ejecuta `hammingnetwork.py` para entrenar la red y procesar las imágenes de prueba.

   ```bash
   python hammingnetwork.py
   ```

   Esto generará el archivo `dataoutput.csv` con los resultados de la clasificación.

2. **Evaluar resultados:**  
   Ejecuta `metrics.py` para obtener métricas de desempeño.

   ```bash
   python metrics.py
   ```

[Repositorio de GitHub](https://github.com/mirkojp/TPFinal/edit/main/README.md)


