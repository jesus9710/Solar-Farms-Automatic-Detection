#%% Libs:

import satlaspretrain_models
import torch
import torchvision.transforms as transforms

from PIL import Image as PIM
import os
from pathlib import Path

weights_manager = satlaspretrain_models.Weights()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# %% Paths:

IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels')

# %% Dataset:

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # Lista de archivos de imagen
        self.masks = os.listdir(mask_dir) # Lista de archivos de máscara

    def __len__(self):

        return len(self.images)

    def __getitem__(self, ix):

        img_path = self.image_dir.joinpath(self.images[ix]) # Archivo de imagen ix
        mask_path = self.mask_dir.joinpath(self.masks[ix]) # Archivo de máscara ix

        image = PIM.open(img_path).convert("RGB")
        mask = PIM.open(mask_path)

        image, mask = self.transform(image).to(device).float(), self.transform(mask).squeeze().to(device).long() # Formato para entrenamiento

        return image, mask

# %% Model

model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinT_SI_RGB", fpn=True, head = satlaspretrain_models.Head.SEGMENT, num_categories=6)
model = model.to(device)

# Congelar pesos del backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# %% Train

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
batch_size = 100
epochs = 5

TRANSFORM = transforms.ToTensor()

dataset = Dataset(IMAGE_DIR, MASK_DIR, TRANSFORM)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

for e in range(1, epochs+1):
    print(f"epoch: {e}/{epochs}")
    for batch_ix, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(x)[0]
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), (batch_ix + 1) * len(x)
        print(f"loss: {loss:.4f} [{current:>5d}/{len(dataset):>5d}]")

# %%
