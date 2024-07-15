# %% Librerías y constantes

import torch # Para el manejo de tensores y operaciones en GPU
import torchvision.transforms as transforms # Para transformaciones de datos
import segmentation_models_pytorch as smp 
from pathlib import Path # Para manejo de rutas de archivos
from utils import * # Funciones y variables del módulo utils

IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels') # Directorio de imágenes (features) y máscaras (labels)
MODEL_DIR = Path.cwd().parent.joinpath('models') # Directorio de modelos

# %% Gestor de pesos y dispositivo de cómputo

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)  # Dispositivo de cómputo: GPU si está disponible, sino CPU

# %% Definición del modelo

load_model = False

if load_model:
    upsample_path = MODEL_DIR / 'upsample_weights.pth'
    head_path = MODEL_DIR / 'head_weights.pth'
else:
    upsample_path = None
    head_path = None

model = Segmentation_model(model_identifier = "Aerial_SwinB_SI",
                           upsample_path=upsample_path,
                           head_path=head_path,
                           num_categories = 2,
                           criterion = 'Dice',
                           device = 'cuda')

model = model.to(device)

# %% Parámetros

batch_size = 40 #  Tamaño del batch (subconjuntos de datos)
epochs = 50 # Número de veces que todo el conjunto de datos es pasado por la red

optimizer = torch.optim.Adam(model.parameters()) # Se elige el optimizador Adam (Adaptive Moment Estimation)
es = EarlyStopping(patience=10)
transform = transforms.ToTensor() # Secuencia de transformaciones antes de enviarlo al modelo donde se convertirá la imagen a tensor para ser procesado en GPU

# %% Train/Test Dataloaders

train_ratio = 0.7
val_ratio = 0.15

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device) # Composición del conjunto de datos

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])  # División del conjunto de datos en entrenamiento y prueba

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size, shuffle=False) # Creación de un DataLoader para el conjunto de entrenamiento, validación y prueba

# %% Train

train_model = True

if train_model:
    hist = model.fit(epochs, optimizer, train_dataloader, val_dataloader, early_stopping=es)
    show_loss_accuracy_evolution(hist)

# %% Predicción

y_hat, soft_preds, images, target = model.predict(test_dataloader)

# %% Evaluación

base_preds = (torch.zeros(*y_hat.shape)).long().to(device) # Baseline (todos los píxeles del valor de la clase mayoritaria)

base_score = evaluate_model(base_preds, target)
score = evaluate_model(y_hat, target)

print(f'BaseLine score: {base_score}')
print(f'model score: {score}')

# %% Visualización aleatoria

ind = np.random.randint(0, len(images))
check_results(images[ind,:,:,:], y_hat[ind,:,:])

# %% Guardar modelo

save_model = False
upsample_fname = 'upsample_weights.pth'
head_fname = 'head_weights.pth'

if save_model:
    upsample_state_dict = model.upsample.state_dict()
    head_state_dict = model.head.state_dict()
    torch.save(upsample_state_dict, MODEL_DIR / upsample_fname)
    torch.save(head_state_dict, MODEL_DIR / head_fname)
