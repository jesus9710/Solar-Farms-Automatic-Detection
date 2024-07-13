#%% Importación de librerías:

import satlaspretrain_models  # Para utilizar modelos preentrenados
import torch # Para el manejo de tensores y operaciones en GPU
import torchvision.transforms as transforms # Para transformaciones de datos
import segmentation_models_pytorch as smp 
from pathlib import Path # Para manejo de rutas de archivos
from utils import * # Funciones y variables del módulo utils

#%% Gestor de pesos y dispositivo de cómputo
weights_manager = satlaspretrain_models.Weights()  # Iniciacilización del gestor de pesos del modelo preentrenado
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)  # Dispositivo de cómputo: GPU si está disponible, sino CPU

# %% Paths:
IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels') # Directorio de imágenes (features) y máscaras (labels)

# %% Model
model = weights_manager.get_pretrained_model( # Obtención del modelo preentrenado 
    model_identifier = "Aerial_SwinB_SI",  # Especificación del identificador del modelo acorde a la capacidad de cómputo y al tipo de input
    fpn = True, # Habilitación de Feature Pyramid Network (FPN) para mejorar la deteccion de objetos de diferentes tamaños 
    head = satlaspretrain_models.Head.SEGMENT,  # Tarea de las capas de cabecera (últimas capas) = segmentación
    num_categories = 2,  # Número de categorías a segmentar
    device = device_str) # Especificación del dispositivo

model = model.to(device)  # Mueve el modelo al dispositivo disponible

#%% Actualización de gradientes (retropropagación)
for param in model.parameters():
    param.requires_grad = False # Congelación de los parámetros del backbone deshabilitando la actualización de sus gradientes para impedir la retropropagación

for param in model.upsample.parameters():
    param.requires_grad = True

for param in model.head.parameters(): 
    param.requires_grad = True # Se habilita la actualización de los gradientes de la cabecera y del upsample para permitir la retropropagación de los errores

# %% Criterio de pérdida
criterion_type = 'Dice' # Selección del criterio de pérdida: 'Dice', 'Focal' o 'CE'

if criterion_type == 'Dice':
    #criterion = GenDiceLoss(eps = 10, device=device)
    criterion = smp.losses.DiceLoss(mode='multiclass', classes=2, eps=1e-07)

elif criterion_type == 'Focal':
    criterion = FocalLoss(alpha= torch.Tensor([1,1,100,1]).to(device), gamma = 1)

elif criterion_type == 'CE':
    criterion = torch.nn.CrossEntropyLoss()

#%% Dataloader
transform = transforms.ToTensor() # Secuencia de transformaciones antes de enviarlo al modelo donde se convertirá la imagen a tensor para ser procesado en GPU

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device) # Composición del conjunto de datos
train_size = int(0.8 * len(dataset)) # Tamaño del conjunto de entrenamiento = 80% 
test_size = len(dataset) - train_size # Tamaño del conjunto de prueba = % restante
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # División del conjunto de datos en entrenamiento y prueba

batch_size = 40 #  Tamaño del batch (subconjuntos de datos)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Creación de un DataLoader para el conjunto de entrenamiento y prueba

# %% Train
epochs = 50 # Número de veces que todo el conjunto de datos es pasado por la red
optimizer = torch.optim.Adam(model.parameters()) # Se elige el optimizador Adam (Adaptive Moment Estimation)

for epoch in range(epochs): # Iteración sobre las épocas
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train() # Configuración del modo entrenamiento
    for batch_ix, (images, masks) in enumerate(train_dataloader): # Iteración sobre los batches
        optimizer.zero_grad() # Reinicio de los gradientes del optimizador
        out_backbone = model.backbone(images) # Envío de las imágenes a través del 'backbone' del modelo para extraer características de bajo y alto nivel 
        out_fpn = model.fpn(out_backbone)  # Envío de las características extraídas por el 'backbone' a través de la FPN (Feature Pyramid Network) para generar mapas de características de múltiples escalas
        out_upsample = model.upsample(out_fpn) # Envío de los mapas de características generados por la FPN a través de la capa de 'upsample' para aumentar la resolución espacial de los mapas de características
        outputs, loss = model.head(0, out_upsample, masks)  # Envío de los mapas de características de alta resolución a través de la 'head' del modelo para obtener las predicciones finales y calcula la pérdida comparando las predicciones con las máscaras reales
        # loss = criterion(outputs, masks)
        loss.backward() # Cálculo de los gradientes de la pérdida con respecto a los parámetros del modelo (retropropagación)
        optimizer.step() # Actualización de los parámetros del modelo utilizando los gradientes calculados y el optimizador 
        loss, current = loss.item(), (batch_ix + 1) * len(images) # Conversión de la pérdida a un valor escalar y visualización del progreso actual en función del número de imágenes procesadas
        print(f"loss ({criterion_type}): {loss:.4f} [{current:>5d}/{len(train_dataset):>5d}]") # Visualización de la pérdida actual y el progreso del entrenamiento 

# %% Prediction

model.eval() # Configuración del modo evaluación

soft_preds = [] # Predicciones suaves
target = [] # Máscaras objetivo
images_list = [] 

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=False) # Creación del dataloader de predicción

with torch.no_grad(): # Desactivación del cálculo de gradientes
    for ind, (image_i, mask_i) in enumerate(train_dataloader): # Iteración sobre los batches
        out_backbone = model.backbone(image_i) # Salida del backbone
        out_fpn = model.fpn(out_backbone) # Salida de FPN
        out_upsample = model.upsample(out_fpn) # Salida de upsample
        print(mask_i.shape)
        outputs, _ = model.head(image_i, out_upsample, mask_i) # Salida de la cabecera del modelo
        soft_preds.append(outputs)
        target.append(mask_i)
        images_list.append(image_i) # Almacenamiento de las predicciones suaves, máscaras objetivo e imágenes

soft_preds = torch.cat(soft_preds)
target = torch.cat(target)
images_ = torch.cat(images_list) # Concatenación de las predicciones suaves, máscaras objetivo e imágenes

# %% Hard output (salida binarizada)

hard_pred = torch.argmax(soft_preds, dim=1) # Obtención de las predicciones duras a partir de las predicciones suaves
print(hard_pred.min()) # Valor mínimo de las predicciones duras

# %% Base Line

base_preds = (torch.zeros(*hard_pred.shape)).long().to(device) # Crea predicciones base (todas ceros)

# %% Evaluation

tp, fp, fn, tn = smp.metrics.get_stats(base_preds, target, mode='multiclass',num_classes=2) 
base_score = smp.metrics.functional.iou_score(tp, fp, fn, tn).mean() # Cálculo de las estadísticas + IoU para la línea base

tp, fp, fn, tn = smp.metrics.get_stats(hard_pred, target, mode='multiclass',num_classes=2)
score = smp.metrics.functional.iou_score(tp, fp, fn, tn).mean() # Cálculo de las estadísticas + IoU para el modelo

print(f'BaseLine score: {base_score}')
print(f'model score: {score}')

# %% Visualization

ind = np.random.randint(0, len(images_))
check_results(images_[ind,:,:,:], hard_pred[ind,:,:])

# %%
