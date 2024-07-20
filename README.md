# TFM-KSchool

Detección automática de granjas solares en imágenes satélite mediante redes neuronales de segmentación

Esta red neuronal realiza tareas de segmentación de imágenes para detectar granjas solares utilizando una arquitectura basada en "Swin Transformers" y "Feature Pyramid Networks". El modelo original fue desarrollado por [Allenai](https://github.com/allenai) y está disponible en [Satlas Pretrain Models](https://github.com/allenai/satlaspretrain_models). Está licenciado bajo la Licencia Apache 2.0. Puedes encontrar más información sobre la licencia en el archivo [LICENSE](./LICENSE.txt).

Las modificaciones realizadas en el presente proyecto sobre el modelo original consisten en un entrenamiento de los bloques de reescalado y segmentación de la arquitectura [Aerial_SwinB_SI](https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth?download=true), manteniendo intactos los pesos del esqueleto ("backbone"). Este entrenamiento se ha llevado a cabo mediante imágenes satelitales de Sentinel 2 de las provincias españolas de Badajoz y Ciudad Real. Adicionalmente, se han empleado datos de uso de suelo obtenidos de la base de datos de [SIOSE Alta Resolución](https://centrodedescargas.cnig.es/CentroDescargas/catalogo.do?Serie=SIOSE) para extraer los correspondeintes polígonos de uso solar.

El proyecto se estructura en tres componentes principales: API Sentinel 2, SIOSE AR y Models.

## Estructura del Proyecto

1. `API Sentinel 2`: descarga y preprocesamiento de datos satelitales de Sentinel 2.
2. `SIOSE AR`: análisis y procesamiento de datos satelitales para la obtencion de máscaras multiclase.
3. `Models`: creación y entrenamiento de la red neuronal de segmentación

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

```powershell
pip install -r requirements.txt
```

## Contacto
Para más información, se puede contactar a los autores del proyecto:
    - jesusferrom97@gmail.com
    - lauramapu95@gmail.com
    - diegotorresmollejo@gmail.com
    - david.garrido.godino@gmail.com
