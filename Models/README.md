# Models

Este módulo contiene scripts y archivos necesarios para la creación y el entrenamiento del modelo de segmentación de imágenes. Utiliza PyTorch y segmentation_models_pytorch para construir el modelo de segmentación basado en el modelo preentrenado `SatlasPretrain`.

## Estructura de la Carpeta
- `data/`: Contiene datos utilizados para entrenamiento y prueba.
  - `features/`: Características (imágenes) para entrenamiento.
  - `labels/`: Etiquetas (máscaras) para entrenamiento.
- `models/`: Almacena los pesos del modelos entrenado.
- `src/`: Scripts para entrenamiento y evaluación del modelo.
  - 'train_model.py': Script principal para el entrenamiento del modelo.
  - 'utils.py': Funciones útiles para el procesamiento de datos.


### Entrenamiento del Modelo
1. Es necesario tener los datos en las carpetas `features/` y `labels/`.

2. Se ejecuta el script `train_model.py` para entrenar el modelo y calcular su precisión.

