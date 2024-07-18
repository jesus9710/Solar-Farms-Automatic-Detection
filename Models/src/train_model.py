# %% Librerías y constantes

import torch # Para el manejo de tensores y operaciones en GPU
import torchvision.transforms as transforms # Para transformaciones de datos
import segmentation_models_pytorch as smp # Para modelos de segmentación
from pathlib import Path # Para manejo de rutas de archivos
from utils import * # Todas las funciones y variables del módulo utils

IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels') # Directorio de imágenes (features) y máscaras (labels)
MODEL_DIR = Path.cwd().parent.joinpath('models') # Directorio de modelos

# %% Gestor de pesos y dispositivo de cómputo

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)  # Dispositivo de cómputo: GPU si está disponible, sino CPU

# %% Definición del modelo

load_model = True # Variable para determinar si se cargan los pesos del modelo preentrenado 

if load_model: # Si se cargan los pesos:
    upsample_path = MODEL_DIR / 'upsample_weights.pth'  # Ruta de los pesos de la capa de upsampling
    head_path = MODEL_DIR / 'head_weights.pth' # Ruta de los pesos de la cabecera del modelo
else: # Si no se cargan los pesos:
    upsample_path = None
    head_path = None

model = Segmentation_model(model_identifier = "Aerial_SwinB_SI",  # Inicialización del modelo de segmentación
                           upsample_path=upsample_path, # Pesos de upsampling
                           head_path=head_path, # Pesos de la cabecera del modelo
                           num_categories = 2, # Número de categorías de segmentación
                           criterion = 'Dice', # Criterio de pérdida
                           device = 'cuda') # Dispositivo de cómputo

model = model.to(device) # Envío del modelo al dispositivo de cómputo

# %% Parámetros

batch_size = 60 #  Tamaño del batch (subconjuntos de datos)
epochs = 200 # Número de veces que todo el conjunto de datos es pasado por la red

optimizer = torch.optim.Adam(model.parameters()) # Elección del optimizador Adam (Adaptive Moment Estimation)
es = EarlyStopping(patience=20) # Configuración de Early Stopping con patience de 20 épocas
transform = transforms.ToTensor() # Secuencia de transformaciones antes de enviarlo al modelo donde se convertirá la imagen a tensor para ser procesado en GPU

# %% Train/Val/Test Dataloaders

train_ratio = 0.7 # Proporción de datos para validación
val_ratio = 0.15 # Proporción de datos para validación

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device) # Creación del conjunto de datos

train_size = int(train_ratio * len(dataset)) # Tamaño del conjunto de entrenamiento
val_size = int(val_ratio * len(dataset))  # Tamaño del conjunto de validación
test_size = len(dataset) - train_size - val_size # Tamaño del conjunto de prueba

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])  # División del conjunto de datos en entrenamiento, validación y prueba

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size, shuffle=False) # Creación de un cargador de datos para el conjunto de entrenamiento, validación y prueba

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_model = False # Variable para determinar si se entrena el modelo

if train_model:
    hist = model.fit(epochs, optimizer, train_dataloader, val_dataloader, early_stopping=es) # Entrenamiento del modelo
    show_loss_accuracy_evolution(hist) # Evolución de  la función de pérdida y de la precisión del modelo

# %% Predicción

y_hat, soft_preds, images, target = model.predict(test_dataloader) # Predicciones del modelo

# %% Evaluación

base_preds = (torch.zeros(*y_hat.shape)).long().to(device) # Predicciones baseline (todos los píxeles del valor de la clase mayoritaria)

base_score = evaluate_model(base_preds, target)  # Evaluación del baseline
score = evaluate_model(y_hat, target) # Evaluación del modelo

print(f'BaseLine score: {base_score}')
print(f'model score: {score}')  # Métricas de evaluación del baseline y del modelo

# %% Visualización aleatoria de resultados

ind = np.random.randint(0, len(images))
check_results(images[ind,:,:,:], y_hat[ind,:,:])

# %% Guardar modelo

save_model = False # Variable para guardar el modelo
upsample_fname = 'upsample_weights.pth' # Archivo donde se almacenan los pesos de upsampling
head_fname = 'head_weights.pth' # Archivo donde se almacenan los pesos de la cabecera del modelo


if save_model:
    upsample_state_dict = model.upsample.state_dict()
    head_state_dict = model.head.state_dict()
    torch.save(upsample_state_dict, MODEL_DIR / upsample_fname)
    torch.save(head_state_dict, MODEL_DIR / head_fname)
