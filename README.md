# TFM-KSchool

Detección automática de granjas solares en imágenes satélite mediante redes neuronales convolucionales (CNN)

Esta red neural convolucional realiza tareas de segmentación de imágenes para detectar granjas solares utilizando los pesos del modelo preentrenado 'SatlasPretrain' sobre imagenes satelitales de Sentinel 2 procesadas.

Se estructura en tres componentes principales: API Sentinel 2, SIOSE AR y Models.

## Estructura del Proyecto

1. 'API Sentinel 2': descarga y preprocesamiento de datos satelitales de Sentinel 2.
2. 'SIOSE AR': análisis y procesamiento de datos satelitales para la obtencion de máscaras multiclase.
3. 'Models': creación y entrenamiento de la red neuronal convolucional (CNN)

## Requisitos

Para ejecutar este proyecto, se necesita tener instalados los siguientes paquetes de Python:

- torchvision
- segmentation_models_pytorch
- pathlib
- numpy
- matplotlib
- rasterio
- geopandas
- fiona
- opencv
- pandas
- pillow
- sentinelhub
- shapely

Se pueden instalar estos paquetes utilizando el archivo 'requirements.txt' proporcionado.

'''bash
pip install -r requirements.txt

## Contacto
Para más información, se puede contactar a los autores del proyecto:
    - jesusferrom97@gmail.com
    - lauramapu95@gmail.com
    - diegotorresmollejo@gmail.com
    - david.garrido.godino@gmail.com
