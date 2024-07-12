import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PIM

class Dataset(torch.utils.data.Dataset):
    """
    Clase Dataset para entrenamiento del modelo de segmentación

    Atributos:
        image_dir (Path): directorio de imágenes RGB
        mask_dir (Path): directorio de máscaras
        transform (transform): transformación de torchvisión

    """

    def __init__(self, image_dir, mask_dir, transform, device = torch.device('cuda')):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(image_dir.glob('*.png')) # Lista de archivos de imagen
        self.masks = list(mask_dir.glob('*.png')) # Lista de archivos de máscara
        self.device = device

    def __len__(self):

        return len(self.images)

    def __getitem__(self, ix):

        img_path = self.images[ix] # Archivo de imagen ix
        mask_path = self.masks[ix] # Archivo de máscara ix
        
        image = PIM.open(img_path).convert("RGB")
        mask = PIM.open(mask_path)

        image, mask = self.transform(image).to(self.device).float(), self.transform(mask).squeeze().to(self.device).long() # Formato para entrenamiento

        return image, mask

class FocalLoss(nn.Module):
    """
    Esta clase implementa una función de pérdida Focal integrable en el ciclo de entrenamiento de pytorch

    Atributos:
        gamma (int): parámetro de ponderación de las muestras mal clasificadas

    Métodos:
        forward(inputs:tensor, targets:tensor):
            Devuelve el valor de la función de pérdida
    """

    def __init__(self, alpha, gamma=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()

        return loss

class GenDiceLoss(nn.Module):
    """
    Esta clase implementa una función de pérdida Dice generalizada integrable en el ciclo de entrenamiento de pytorch

    Atributos:
        eps (float): constante de volumen (preferiblemente >= 1)

    Métodos:
        forward(inputs:tensor, targets:tensor):
            Devuelve el valor de la función de pérdida
    """

    def __init__(self, eps = 10, device = torch.device('cuda')):
        super(GenDiceLoss, self).__init__()
        self.eps = eps
        self.device = device

    def forward(self, inputs, targets):
        inputs = inputs.log_softmax(dim=1).exp()
        targets_ohe = F.one_hot(targets)
        wl = 1 / ((targets_ohe.sum(dim=(0)) + self.eps) ** 2)

        num = torch.zeros(targets_ohe.shape).to(self.device)
        den = torch.zeros(targets_ohe.shape).to(self.device)

        for i in range(num.shape[-1]):
            num[:,:,:,i] = targets_ohe[:,:,:,i] * inputs[:,i,:,:]
            den[:,:,:,i] = targets_ohe[:,:,:,i] + inputs[:,i,:,:]

        num = num.sum(dim=(0))
        num = (num * wl).sum(dim=-1)

        den = den.sum(dim=(0))
        den = (den * wl).sum(dim=-1)

        loss = (1 - 2 * num/den).mean()
        return loss
    
def calculate_weights_FLoss (mask_path, n_classes, n_dim, device):
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
    total_samples = len(mask_files) * (n_dim ** 2)

    for file in mask_files:
        mask = np.array(PIM.open(file))
        class_counts = np.bincount(mask.flatten())
        px_count += class_counts
    
    class_weights = []
    for count in px_count:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)
        
    return torch.FloatTensor(class_weights).to(device)
