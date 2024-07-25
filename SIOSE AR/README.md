# SIOSE AR

Este módulo contiene scripts de procesamiento de datos satelitales y análisis de uso del suelo para la creación de máscaras espaciales multiclase.

## Estructura de la Carpeta
- `spatial_data/`: Almacena datos espaciales.
- `01 - merging and uses extraction.ipynb`: Notebook para la fusión y extracción de usos del suelo.
- `02 - mask sentinel by polygons.ipynb`: Notebook para la creación de máscaras basadas en polígonos.
- `03 - Raster and multiclass mask split.ipynb`: Notebook para la división de máscaras rasterizadas en múltiples clases.

### Análisis de Datos Espaciales
1. Deben ejecutarse los notebooks en el orden proporcionado para realizar el análisis completo y la creación de máscaras.

2. Los datos de entrada y salida se almacenan en la carpeta `spatial_data/`.

3. Los datos necesarios para la ejecución del código se encuentran disponbibles para descarga en el Centro de Descargas del CNIG, además de las imágenes descargadas previamente en la carpeta 'API Sentinel 2'.
