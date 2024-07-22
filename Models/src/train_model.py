# %% Librerías y constantes

import torch # Para el manejo de tensores y operaciones en GPU
import torchvision.transforms as transforms # Para transformaciones de datos
from pathlib import Path # Para manejo de rutas de archivos
from utils import * # Todas las funciones y variables del módulo utils

IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels') # Directorio de imágenes (features) y máscaras (labels)
MODEL_DIR = Path.cwd().parent.joinpath('models/GDLv/Aerial Swin Base') # Directorio de modelos

# %% Gestor de pesos y dispositivo de cómputo

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)  # Dispositivo de cómputo: GPU si está disponible, sino CPU

# %% Definición del modelo

load_model = True # Variable para determinar si se cargan los pesos del modelo preentrenado 

if load_model: # Si se cargan los pesos:
    backbone_path = MODEL_DIR / 'BB_ASw_GDLv.pth'
    upsample_path = MODEL_DIR / 'UP_ASw_GDLv.pth'
    head_path = MODEL_DIR / 'HE_ASw_GDLv.pth'
else: # Si no se cargan los pesos:
    backbone_path = None
    upsample_path = None
    head_path = None

model = Segmentation_model(model_identifier = "Aerial_SwinB_SI", # Inicialización del modelo de segmentación
                           backbone_path=backbone_path, # Pesos del backbone
                           upsample_path=upsample_path, # Pesos de upsampling
                           head_path=head_path, # Pesos de la cabecera del modelo
                           num_categories = 2, # Número de categorías de segmentación
                           criterion = 'GDLv', # Criterio de pérdida
                           device = 'cuda') # Dispositivo de cómputo

model = model.to(device) # Envío del modelo al dispositivo de cómputo

# %% Parámetros

batch_size = 40 #  Tamaño del batch (subconjuntos de datos)
epochs = 200 # Número de veces que todo el conjunto de datos es pasado por la red

optimizer = torch.optim.Adam(model.parameters()) # Elección del optimizador Adam (Adaptive Moment Estimation)
es = EarlyStopping(patience=10, min_delta=0.02) # Configuración de Early Stopping con patience de 10 épocas

# Data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3), # Volteo hotizontal
    transforms.RandomVerticalFlip(p=0.3) # Volteo vertical
])

# RandomTransform aplica la misma transformación con misma semilla para imágenes y máscaras
transform = RandomTransform(transform)

# %% Train/Val/Test Dataloaders

train_ratio = 0.7 # Proporción de datos para validación
val_ratio = 0.15 # Proporción de datos para validación

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device) # Creación del conjunto de datos

train_size = int(train_ratio * len(dataset)) # Tamaño del conjunto de entrenamiento
val_size = int(val_ratio * len(dataset))  # Tamaño del conjunto de validación
test_size = len(dataset) - train_size - val_size # Tamaño del conjunto de prueba

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(42))  # División del conjunto de datos en entrenamiento y prueba

# Para test no es necesario data augmentation
test_dataset.transform = None

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size, shuffle=False) # Creación de un cargador de datos para el conjunto de entrenamiento, validación y prueba

# %% Training

train_model = False # Variable para determinar si se entrena el modelo

if train_model:
    hist = model.fit(epochs, optimizer, train_dataloader, val_dataloader, early_stopping=es) # Entrenamiento del modelo
    show_loss_accuracy_evolution(hist) # Evolución de  la función de pérdida y de la precisión del modelo

# %% Predicción

y_hat_train, soft_preds_train, _, target_train = model.predict(train_dataloader)
y_hat_val, soft_preds_val, _, target_val = model.predict(val_dataloader)
y_hat, soft_preds, images, target = model.predict(test_dataloader) # Predicciones del modelo

# %% Evaluación (threshold = 0.5)

base_preds = (torch.ones(*y_hat.shape)).long().to(device) # Predicciones baseline (todos los píxeles del valor de la clase mayoritaria)

base_score = evaluate_model(base_preds.unsqueeze(1), target.unsqueeze(1)) # Evaluación del baseline
train_score = evaluate_model(y_hat_train.unsqueeze(1), target_train.unsqueeze(1)) 
val_score = evaluate_model(y_hat_val.unsqueeze(1), target_val.unsqueeze(1))
score = evaluate_model(y_hat.unsqueeze(1), target.unsqueeze(1)) # Evaluación del modelo

print(f'BaseLine score: {base_score}')
print(f'model score (train): {train_score}')
print(f'model score (val): {val_score}')
print(f'model score (test): {score}') # Métricas de evaluación del baseline y del modelo

# %% Evaluación AUC

base_preds = (torch.ones(*soft_preds.shape)).long().to(device) # Baseline (todos los píxeles del valor de la clase mayoritaria)

base_score, _, _ = evaluate_model_auc(base_preds, target)
train_score, _, _ = evaluate_model_auc(soft_preds_train, target_train)
val_score, _, _ = evaluate_model_auc(soft_preds_val, target_val)
score, thresholds, iou_socres = evaluate_model_auc(soft_preds, target)

print(f'BaseLine score: {base_score}')
print(f'model score (train): {train_score}')
print(f'model score (val): {val_score}')
print(f'model score (test): {score}')

show_IOU_curve(thresholds, iou_socres)

# %% Visualización aleatoria de resultados

# Índice aleatorio
ind = np.random.randint(0, len(images))

th = 0.5

if th:
    # Los píxeles solares son únicamente aquellos con th% de probabilidad
    predictions = torch.where(soft_preds[:,1:2,:,:].squeeze(dim=1) >= th, 1, 0)

else:
    # Las predicciones son la clase con mayor probabilidad
    predictions = y_hat

check_results(images[ind,:,:,:], predictions[ind,:,:], target[ind,:,:])

# %% Guardar modelo

save_model = False # flag de seguridad para guardar modelo

backcone_fname = 'BB_ASw_GDLv.pth' # Archivo donde se almacenan los pesos de backbone
upsample_fname = 'UP_ASw_GDLv.pth' # Archivo donde se almacenan los pesos de upsampling
head_fname = 'HE_ASw_GDLv.pth' # Archivo donde se almacenan los pesos de la cabecera del modelo

if save_model:
    backbone_state_dict = model.backbone.state_dict()
    upsample_state_dict = model.upsample.state_dict()
    head_state_dict = model.head.state_dict()
    torch.save(backbone_state_dict, MODEL_DIR / backcone_fname)
    torch.save(upsample_state_dict, MODEL_DIR / upsample_fname)
    torch.save(head_state_dict, MODEL_DIR / head_fname)
    
#%% Predicción sobre imágenes nuevas (Badajoz, Junio 2024) sin labels

def main():
    
    data_dir = 'data/predict' # Directorio que contiene las imágenes
    img_paths = [str(path) for path in Path.cwd().parent.joinpath(data_dir).glob('*.png')] # Lista de rutas completas de las imágenes en el directorio

    prediction = PredDatasetWithPrediction(image_dir=data_dir, model=model) # Crear una instancia de la clase, pasando el directorio de imágenes y el modelo

    for img_path in img_paths: # Iterar sobre la lista de imágenes y realizar la predicción usando la instancia
        prediction.predict_single_image(img_path, output_dir='../results')

if __name__ == "__main__":
    main()

# %%
