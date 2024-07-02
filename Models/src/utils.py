import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PIM

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
    """
    Clase Dataset para entrenamiento del modelo de segmentación

    Atributos:
        image_dir (Path): directorio de imágenes RGB
        mask_dir (Path): directorio de máscaras
        transform (transform): transformación de torchvisión

    """

    def __init__(self, image_dir, mask_dir, transform):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(image_dir.glob('*.png')) # Lista de archivos de imagen
        self.masks = list(mask_dir.glob('*.png')) # Lista de archivos de máscara

    def __len__(self):

        return len(self.images)

    def __getitem__(self, ix):

        img_path = self.image_dir.joinpath(self.images[ix]) # Archivo de imagen ix
        mask_path = self.mask_dir.joinpath(self.masks[ix]) # Archivo de máscara ix

        image = PIM.open(img_path).convert("RGB")
        mask = PIM.open(mask_path)

        image, mask = self.transform(image).to(device).float(), self.transform(mask).squeeze().to(device).long() # Formato para entrenamiento

        return image, mask

class FocalLoss(nn.Module):
    """
    Esta clase implementa una función de pérdida Focal integrable en el ciclo de entrenamiento
    de pytorch

    Attributes:
        alpha (tensor): tensor de pesos asociados a cada clase
        gamma (int): parámetro de ponderación de las muestras mal clasificadas

    Methods:
        forward(inputs:tensor, targets:tensor):
            Devuelve el valor de la función de pérdida
    """

    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

def calculate_weights_FLoss (mask_path, n_classes, n_dim):
    """
    Función para calcular los pesos necesarios para la función de pérdida focal

    Args:
        mask_path (Path): directorio de máscaras
        n_classes (int): número de clases
        n_dim (int): tamaño de la imagen (lado del cuadrado)

    Returns:
        tensor: pesos necesarios para la función de pérdida focal
    """

    mask_files = list(mask_path.glob('*.png'))
    px_count = np.zeros(n_classes)

    total_samples = 1
    for i in (len(mask_files), n_dim, n_dim):
        total_samples *= i

    for file in mask_files:
        mask = np.array(PIM.open(file))
        class_counts = np.bincount(mask.flatten())
        px_count += class_counts
    
    class_weights = []
    for count in px_count:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)
        
    return torch.FloatTensor(class_weights).to(device)
