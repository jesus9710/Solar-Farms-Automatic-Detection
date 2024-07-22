import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc
from PIL import Image as PIM
import matplotlib.pyplot as plt
import satlaspretrain_models
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import random

class Dataset(torch.utils.data.Dataset):
    """
    Clase Dataset para entrenamiento del modelo de segmentación

    Atributos:
        image_dir (Path): directorio de imágenes RGB
        mask_dir (Path): directorio de máscaras
        transform (transform): transformación de torchvisión
        images (list): lista de archivos de imagen
        masks (list): lista de archivos de máscara
        device (device): dispositivo de computación
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

        data_format = transforms.ToTensor()

        image, mask = data_format(image).to(self.device).float(), data_format(mask).squeeze().to(self.device).long() # Formato para entrenamiento

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
    
class Segmentation_model(nn.Module):
    """
    Modelo de Segmentación

    Atributos:
        criterion (torch loss): Función de pérdida
        criterion_type (str): Tipo de función de pérdida
        num_categories (int): Número de categorías
        mode (str): Determina cómo se va a evaluar el modelo
            - "binary": Los tensores son de dimensión (N,1,...). Se obtienen métricas a partir de la clase 1
            - "multiclass": Los tensores son de dimensión (N,...). Se obtienen métricas promediando todas las clases
        eval_with_clases (int): None si el modo es binario. En caso contrario es igual al número de clases
            
    Métodos:
        forward(x:tensor, y:tensor):
            Propagación hacia adelante de la red neuronal
        fit(epochs: int, optimizer: torch optimizer, train_dataloader: torch dataloader,
            val_dataloader: torch dataloader, early_stopping: Early_Stopping)
        predict(dataloader:torch dataloader, threshold:float):
            Obtención de predicciones
        evaluate_model(y_hat: tensor, y: tensor): 
    """

    def __init__(self, model_identifier, backbone_path = None, upsample_path = None, head_path = None, num_categories = 2, criterion = 'CrossEntropy', alpha = None, gamma = None, device = 'cuda'):
        """
        Inicializa una nueva instancia de Segmentation_model

        Args:
            model_identifier (str): Identificador del modelo de "Satlas Pretrain Models"
            backbone_path (path): Ubicación del archivo .pth que almacena los pesos del bloque backbone
            upsample_path (path): Ubicación del archivo .pth que almacena los pesos del bloque upsample
            head_path (path): Ubicación del archivo .pth que almacena los pesos del bloque head
            num_categories (int): Número de categorías del problema de segmentación (debe ser >2)
            criterion (str): Tipo de función de pérdida. Están disponibles las siguientes:
                - "CrossEntropy": Entropía Cruzada
                - "Focal": Función de pérdida Focal
                - "Dice": Función de pérdida Dice
                - "GDLv": Función de pérdida Dice con ponderación de clases basado de volumen de píxeles
            alpha (list): (Sólo con Focal Loss) Lista de pesos de ponderación de clases. Cada valor es un float de 0 a 1
                La suma de todos los elementos debe ser igual a 1. El tamaño de la lista debe ser igual al número de categorías
            gamma (int): (Sólo con Focal Loss) Factor de modulación (ponderación de las muestras mal clasificadas)
            device (str): Dispositivo de cómputo ("cpu" o "cuda")
        """
        super(Segmentation_model, self).__init__()

        # Obtención del modelo fundacional
        weights_manager = satlaspretrain_models.Weights()
        model = weights_manager.get_pretrained_model( # Obtención del modelo preentrenado
            model_identifier = model_identifier,  # Especificación del identificador del modelo acorde a la capacidad de cómputo y al tipo de input
            fpn = True, # Habilitación de Feature Pyramid Network (FPN) para mejorar la deteccion de objetos de diferentes tamaños
            head = satlaspretrain_models.Head.SEGMENT,  # Tarea de las capas de cabecera (últimas capas) = segmentación
            num_categories = num_categories,  # Número de categorías a segmentar
            device = device) # Especificación del dispositivo

        self.__dict__ = model.__dict__.copy()
        self.num_categories = num_categories

        # Si se tienen más de dos clases, las métricas se calcularán en formato multiclase
        if self.num_categories > 2:
            self.mode = "multiclass"
            self.eval_with_classes = self.num_categories
        else:
            self.mode = "binary"
            self.eval_with_classes = None

        # Cargar bloques de la red
        if backbone_path:
            self._Load_Backbone(backbone_path)
        
        if upsample_path:
            self._Load_Upsample(upsample_path)

        if head_path:
            self._Load_Head(head_path)

        # Congelar pesos del bloque backbone
        for param in self.parameters():
            param.requires_grad = False # Congelación de los parámetros del backbone deshabilitando la actualización de sus gradientes para impedir la retropropagación

        for param in self.upsample.parameters():
            param.requires_grad = True

        for param in self.head.parameters(): 
            param.requires_grad = True # Se habilita la actualización de los gradientes de la cabecera y del upsample para permitir la retropropagación de los errores

        # Selección de función de pérdida
        if criterion == 'Dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass', classes=self.num_categories, eps=1e-07)
            self.criterion_type = 'Dice'
        
        elif criterion == 'GDLv':
            self.criterion = GenDiceLoss(eps = 10, device=device)
            self.criterion_type = 'GDLv'

        elif criterion == 'Focal':
            self.criterion = FocalLoss(alpha= torch.Tensor(alpha).to(device), gamma = gamma)
            self.criterion_type = 'Focal Loss'

        elif criterion == 'CrossEntropy':
            self.criterion = None
            self.criterion_type = 'CrossEntropy'
        
        else:
            self.criterion = None
            self.criterion_type = 'CrossEntropy'
        
    def forward(self, x, y = None):

        out_backbone = self.backbone(x) # Envío de las imágenes a través del 'backbone' del modelo para extraer características de bajo y alto nivel 
        out_fpn = self.fpn(out_backbone)  # Envío de las características extraídas por el 'backbone' a través de la FPN (Feature Pyramid Network) para generar mapas de características de múltiples escalas
        out_upsample = self.upsample(out_fpn) # Envío de los mapas de características generados por la FPN a través de la capa de 'upsample' para aumentar la resolución espacial de los mapas de características
        outputs, _ = self.head(0, out_upsample, y)  # Envío de los mapas de características de alta resolución a través de la 'head' del modelo para obtener las predicciones finales y calcula la pérdida comparando las predicciones con las máscaras reales
        
        if y is not None and self.criterion:
            loss = self.criterion(outputs, y)
        else:
            loss = None

        return outputs, loss
    
    def fit(self, epochs, optimizer, train_dataloader, val_dataloader = None, early_stopping = None):
        """
        Realiza un bucle de entrenamiento del modelo de segmentación

        Args:
            epochs (int): Número de épocas
            optimizer (torch optimizer): Algoritmo de optimización
            train_dataloader (torch dataloader): dataloader de entrenamiento
            val_dataloader (torch dataloader): dataloader de validación
            early_stopping (EarlyStopping): algoritmo de early stopping
        """
        hist = History(loss=self.criterion_type)

        # Bucle de entrenamiento
        for epoch in range(epochs): # Iteración sobre las épocas
            self.train() # Configuración del modo entrenamiento
            loss_hist = []

            # Actualización del gradiente
            for images, masks in train_dataloader: # Iteración sobre los batches
                optimizer.zero_grad() # Reinicio de los gradientes del optimizador
                outputs, loss = self(images, masks)
                if self.criterion:
                    loss = self.criterion(outputs, masks)
                loss.backward() # Cálculo de los gradientes de la pérdida con respecto a los parámetros del modelo (retropropagación)
                optimizer.step() # Actualización de los parámetros del modelo utilizando los gradientes calculados y el optimizador 
                loss_hist.append(loss.item())

            loss = np.array(loss_hist).mean()

            # Cálculo de métrica
            y_hat, *_, target = self.predict(train_dataloader)
            iou = self.evaluate_model(y_hat, target)

            # Pérdida y métrica de validación
            if val_dataloader:
                val_loss_hist = []

                for images, masks in val_dataloader:
                    _, val_loss_i = self(images, masks)
                    val_loss_hist.append(val_loss_i.item())
                
                val_loss = np.array(val_loss_hist).mean()

                y_hat, *_, target = self.predict(val_dataloader)
                val_iou = self.evaluate_model(y_hat, target)

                # Actualización del historial (con métrica de validación)
                hist.update(epoch, loss, iou, val_loss, val_iou)
                print(f"Epoch: [{epoch+1}/{epochs}] | train_loss({self.criterion_type}): {loss:.4f} | val_loss({self.criterion_type}): {val_loss:.4f} | train_iou: {iou:.4f} | val_iou: {val_iou:.4f}")
            
            else:
                # Actualización del historial (sin métrica de validación)
                hist.update(epoch, loss, iou)
                print(f"Epoch: [{epoch+1}/{epochs}] | train_loss({self.criterion_type}): {loss:.4f} | train_iou: {iou:.4f}")
            
            # Early stopping
            if early_stopping:
                early_stopping((self.upsample, self.head),val_loss)
                
                if early_stopping.early_stop:
                    w_upsample, w_head = early_stopping.best_weigths
                    self.upsample.load_state_dict(w_upsample)
                    self.head.load_state_dict(w_head)
                    print("Early stopping")
                    early_stopping._reset()

                    break

        print("Fin del entrenamiento")

        return hist.get_history()
    
    def predict(self, dataloader, threshold = None):
        """
        Realiza una predicción a partir de un dataloader

        Args:
            dataloader (torch dataloader): dataloader que se utilizará para la predicción
            threshold (float): umbral para la predicción (valor entre 0 y 1)

        Returns:
            tuple: tupla con predicciones y las imágenes y máscaras utilizadas:
                - y_hat (tensor): predicción basada en el argumento que maximiza la salida del modelo en el eje clase
                - soft_preds (tensor): salida del modelo
                - images (tensor): tensor que contiene las imágenes utilizadas para predecir
                - target (tensor): tensor que contiene las máscaras utilizadas para predecir
        """

        self.eval()

        soft_preds = [] # Predicciones suaves
        target = [] # Máscaras objetivo
        images_list = [] 

        with torch.no_grad(): # Desactivación del cálculo de gradientes
            for image_i, mask_i in dataloader: # Iteración sobre los batches
                outputs, _ = self(image_i, mask_i) # Salida de la cabecera del modelo
                soft_preds.append(outputs)
                target.append(mask_i)
                images_list.append(image_i) # Almacenamiento de las predicciones suaves, máscaras objetivo e imágenes

        soft_preds = torch.cat(soft_preds)
        target = torch.cat(target)
        images = torch.cat(images_list) # Concatenación de las predicciones suaves, máscaras objetivo e imágenes

        # Aplicar umbral (solo para num_categories = 2)
        if threshold:
            y_hat = torch.where(soft_preds[:,1:2,:,:].squeeze(dim=1) >= threshold, 1, 0) # clasificación por umbral de la clase 1
        else:
            y_hat = torch.argmax(soft_preds, dim=1) # Argumento que maximiza la dimensión clase

        return y_hat, soft_preds, images, target
    
    def evaluate_model(self, y_hat, y):
        """
        Realiza una evaluación de la métrica IOU

        Args:
            y_hat (tensor): predicciones
            y (tensor): valores reales

        Returns:
            float: valor IOU
        """

        if self.eval_with_classes is None:
            y_hat = y_hat.unsqueeze(1)
            y = y.unsqueeze(1)

        return evaluate_model(y_hat, y, self.mode, self.eval_with_classes)

    def _Load_Head(self, head_path):
        head_state_dict = torch.load(head_path)
        self.head.load_state_dict(head_state_dict)

    def _Load_Upsample(self, upsample_path):
        upsample_state_dict = torch.load(upsample_path)
        self.upsample.load_state_dict(upsample_state_dict)

    def _Load_Backbone(self, backbone_path):
        backbone_state_dict = torch.load(backbone_path)
        self.backbone.load_state_dict(backbone_state_dict)

class History:
    """
    Historial para entrenamiento de modelo de segmentación

    Atributos:
        history (dict): historial
    
    Métodos:
        update(epoch(int), train_loss(float), train_metric(float), val_loss:(float), val_metric:(float)):
            actualización dedl historial
        get_history():
            Obtención del historial
    """
    def __init__(self, loss = 'CrossEntropy'):
        """
        Inicializa una nueva instancia de History

        Args:
            loss (str): Nombre de la función de pérdida utilizada
        """
        self.history = {'epoch':[],'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': [], 'loss': loss}

    def update(self, epoch, train_loss, train_metric, val_loss = None, val_metric = None):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metric'].append(train_metric)
        self.history['val_metric'].append(val_metric)

    def get_history(self):
        return self.history

class EarlyStopping:
    """
    Early Stopping para el entrenamiento del modelo de segmentación

    Atributos:
        patience (dict): cantidad de epochs para activar el early stopping
        min_delta (float): mejora mínima para resetear el contador
        best_loss (float): registro del mejor valor de la función de pérdida
        counter (int): contador de número de epochs en los que no hay mejora
        early_stop (boolean): Indicador de activación del early stopping
        best_weigths (tuple): Tupla que contiene el mejor modelo (upsample.state_dict(), head.state_dict())
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Inicializa una nueva instancia de EarlyStopping

        Args:
            patience (dict): cantidad de epochs para activar el early stopping
            min_delta (float): mejora mínima para resetear el contador
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_weigths = None

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weigths = tuple(block.state_dict() for block in model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weigths = tuple(block.state_dict() for block in model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _reset(self):
        self.counter = 0
        self.early_stop = False
    
    def _hard_reset(self):
        self.counter = 0
        self.early_stop = False
        self.best_loss = None
        self.best_weigths = None

class RandomTransform:
    """
    Esta clase permite aplicar una composición de transformaciones con componente aleatorio y misma semilla
    para imagen y máscara.

    Atributos:
        transform (torch transform): transformación de torch o torchvision con componente aleatorio
    """
    def __init__(self, transform):
        """
        Inicializa una nueva instancia de RandomTransform

        Args:
            transform (torch transform): transformación de torch o torchvision con componente aleatorio
        """
        self.transform = transform

    def __call__(self, image, mask):
        # Generar una semilla aleatoria
        seed = np.random.randint(0, 2**31)
        
        # Transformar imagen
        random.seed(seed)
        torch.manual_seed(seed)
        transformed_image = self.transform(image)

        # Transformar máscara
        random.seed(seed)
        torch.manual_seed(seed)
        transformed_mask = self.transform(mask)
        
        return transformed_image, transformed_mask
    
class FocalLoss(nn.Module):
    """
    Esta clase implementa una función de pérdida Focal integrable en el ciclo de entrenamiento de pytorch

    Atributos:
        alpha (list): Lista de pesos de ponderación de clases. Cada valor es un float de 0 a 1.
            La suma de todos los elementos debe ser igual a 1. El tamaño de la lista debe ser igual al número de categorías.
        gamma (int): factor de modulación (ponderación de las muestras mal clasificadas)

    Métodos:
        forward(inputs:tensor, targets:tensor):
            Devuelve el valor de la función de pérdida
    """

    def __init__(self, alpha, gamma=1):
        """
        Inicializa una nueva instancia de FocalLoss

        Args:
            alpha (list): Lista de pesos de ponderación de clases. Cada valor es un float de 0 a 1.
                La suma de todos los elementos debe ser igual a 1. El tamaño de la lista debe ser igual al número de categorías.
            gamma (int): factor de modulación (ponderación de las muestras mal clasificadas)
        """
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
    Esta clase implementa una función de pérdida Dice generalizada con ponderación de clases
    basado en volumen de píxeles, integrable en el ciclo de entrenamiento de pytorch

    Atributos:
        eps (float): constante de volumen (preferiblemente >= 1)

    Métodos:
        forward(inputs:tensor, targets:tensor):
            Devuelve el valor de la función de pérdida
    """

    def __init__(self, eps = 10, device = torch.device('cuda')):
        """
        Inicializa una nueva instancia de GenDiceLoss

        Args:
            eps (float): constante de volumen (preferiblemente >= 1)
            device (torch device): dispositivo de cómputo
        """
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

def check_results(image, hard_pred, target):
    """
    Función para graficar imágenes con una capa superpuesta de predicciones

    Args:
        image (tensor): imagen RGB (Valores normalizados entre 0-1)
        hard_pred (tensor): máscara de predicciones
        target (tensor): máscara de valores reales
    """
    array_base = image.permute(1,2,0).cpu().numpy()
    array_preds = hard_pred.cpu().numpy()

    # Crear una copia de la imagen para superponer la matriz
    preds_image = array_base.copy()

    # Crear una máscara para los píxeles predichos como solares
    predicciones = array_preds == 1

    # Aplicar color amarillo a los píxeles solares
    preds_image[predicciones] = [255, 255, 0]

    # Mismo proceso para el target
    array_mask = target.cpu().numpy() == 1
    target_image = array_base.copy()
    target_image[array_mask] = [0, 128, 255]

    # Mostrar la imagen original y las máscaras superpuestas
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(array_base)
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')

    axes[1].imshow(preds_image)
    axes[1].set_title("Predicciones")
    axes[1].axis('off')

    axes[2].imshow(target_image)
    axes[2].set_title("Máscara Original")
    axes[2].axis('off')

    plt.show()

def evaluate_model(predictions, y, mode = 'binary', num_classes = None):
    """
    Función para evaluar el IOU

    Args:
        predictions (tensor): tensor de predicciones
        y (tensor): tensor "ground truth"
        mode (str): determina el formato de los tensores:
            - modo 'binary': (N,1,...)
            - modo 'multilabel': (N,C,...)
            - modo 'multiclass': (N,...)
        num_classes (int): número de clases (para modo 'multiclass')
    
    Returns:
        float: Métrica IOU
    """
    
    tp, fp, fn, tn = smp.metrics.get_stats(predictions, y, mode=mode, num_classes=num_classes)
    score = smp.metrics.functional.iou_score(tp, fp, fn, tn).mean() # Cálculo de las estadísticas + IoU para el modelo
    return float(score)

def evaluate_model_auc(predictions, y, negative_preds = False):
    """
    Función para evaluar el área bajo la curva de la métrica IOU a partir de una predicción en formato multiclase con dos categorías.
    La predicción en formato multiclase (N,...) debe estar compuesta por dos categorías.

    Args:
        predictions (tensor): tensor de predicciones "suaves" (N,...)
        y (tensor): tensor "ground truth" (N,...)
        negative_preds (bool): indica cuál es la clase que se quiere evaluar (False: clase 1)
    
    Returns:
        tuple: tupla con métrica AUC IOU y valores de umbrales e IOU:
            - auc_score: Valor AUC IOU (float)
            - thresholds: lista de umbrales (list)
            - scores: lista de valores IOU (list)
    """
    # Vector de threshold (21 puntos)
    thresholds = np.linspace(0, 1, 21)
    scores = []

    # Obtenemos las probabilidades correspondientes a la clase 1
    if negative_preds:
        predictions = predictions[:,0:1,:,:]
    else:
        predictions = predictions[:,1:2,:,:]
    
    y = y.unsqueeze(1) # Mismas dimensiones que las predicciones

    # Obtención de la métrica IOU para cada valor de threshold 
    for threshold in thresholds:

        y_hat = torch.where(predictions >= threshold, 1, 0)

        score = evaluate_model(y_hat, y, mode='binary')
        scores.append(score)

    return auc(thresholds, scores), thresholds, scores

def show_loss_accuracy_evolution(history):
    """
    Función para graficar la evolución de la función de pérdida y de la métrica

    Args:
        history (History): Historial de entrenamiento del modelo de segmentación
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(history['loss'])
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss')
    ax1.plot(history['epoch'], history['val_loss'], label = 'Val Loss')
    ax1.grid()
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IOU')
    ax2.plot(history['epoch'], history['train_metric'], label='Train IOU')
    ax2.plot(history['epoch'], history['val_metric'], label = 'Val IOU')
    ax2.grid()
    ax2.legend()

    plt.show()

def show_IOU_curve(thresholds, scores):
    """
    Función para graficar la evolución de la curva IOU

    Args:
        thresholds (list): lista de umbrales (float)
        scores (list): valores IOU (float)
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel('Threshold')
    ax.set_ylabel('IOU')
    ax.plot(thresholds,scores)
    ax.grid()

    plt.show()
    
class PredDatasetWithPrediction:
    """
    Cargar imágenes RGB una a una desde path, transformar a tensor, predecir sin labels, 
    y guardar en carpeta 'results'. De esta forma liberamos RAM.

    Atributos:
        image_dir (Path): Directorio de imágenes RGB
        transform (transform): Transformación de torchvisión
        model (nn.Module): Modelo de predicción
        device (device): Dispositivo de computación
    """

    def __init__(self, image_dir, model, transform=None, device=None):
        self.image_dir = Path(image_dir)
        self.transform = transform if transform else transforms.ToTensor()
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict_single_image(self, image_path, output_dir='results'):
        """
        Realiza una predicción sobre una sola imagen y guarda el resultado.
        Igual que la función 'predict' pero sin dataloader, directamente sobre RGB.
        """
        self.model.eval()

        image = PIM.open(image_path).convert("RGB")

        if self.transform:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
        else:
            raise ValueError("Transformation not defined")

        with torch.no_grad():
            outputs, _ = self.model(image_tensor)
            soft_preds = outputs.squeeze(0) 
            y_hat = torch.argmax(soft_preds, dim=0)
            y_hat_image = self._convert_to_image(y_hat.cpu())

        # Crear directorio de salida si no existe
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extraer el nombre del archivo de entrada para generar el nombre de salida
        base_name = Path(image_path).name
        pred_image_path = output_path / f'pred_{base_name}'

        # Guardar
        y_hat_image.save(pred_image_path)
        print(f'Predicción guardada en: {pred_image_path}')

        return pred_image_path  # Devolver la ruta

    def _convert_to_image(self, tensor):
        """
        Convierte un tensor de predicciones en una imagen.
        """
        # Convertir el tensor a un numpy array y escalar los valores a 0-255
        tensor_np = tensor.numpy().astype(np.uint8)

        # Crear una imagen PIL a partir del array numpy
        return PIM.fromarray(tensor_np, mode='L')  # 'L' para imágenes en escala de grises