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

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Train
epochs = 1
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    for batch_ix, (images, masks) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images)[0]
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch_ix + 1) * len(x)
        print(f"loss ({criterion_type}): {loss:.4f} [{current:>5d}/{len(dataset):>5d}]")

print('Fin modelo actual desarrollado ')
