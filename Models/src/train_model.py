#%% Libs:

import satlaspretrain_models
import torch
import torchvision.transforms as transforms

from pathlib import Path

from utils import *

weights_manager = satlaspretrain_models.Weights()
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# %% Paths:

IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels')

# %% Model

model = weights_manager.get_pretrained_model(
    model_identifier = "Sentinel2_SwinT_SI_RGB",
    fpn = True,
    head = satlaspretrain_models.Head.SEGMENT,
    num_categories = 6, 
    device = device_str)

model = model.to(device)

# Congelar pesos del backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# %% Criterion

criterion_type = 'Dice' # Seleccionar entre 'Dice', 'Focal' o 'CE'

if criterion_type == 'Dice':
    criterion = GenDiceLoss(eps = 10, device=device)

elif criterion_type == 'Focal':
    criterion = FocalLoss(gamma = 1)

elif criterion_type == 'CE':
    criterion = torch.nn.CrossEntropyLoss()

# %% Train

batch_size = 10
epochs = 1

optimizer = torch.optim.Adam(model.parameters())
TRANSFORM = transforms.ToTensor()

dataset = Dataset(IMAGE_DIR, MASK_DIR, TRANSFORM, device)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

for e in range(1, epochs+1):
    print(f"epoch: {e}/{epochs}")
    for batch_ix, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(x)[0]
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), (batch_ix + 1) * len(x)
        print(f"loss ({criterion_type}): {loss:.4f} [{current:>5d}/{len(dataset):>5d}]")

# %%
print('fin')