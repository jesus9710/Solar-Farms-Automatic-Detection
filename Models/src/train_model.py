#%% Libs:

import satlaspretrain_models
import torch
import torchvision.transforms as transforms

from pathlib import Path

from utils import *

weights_manager = satlaspretrain_models.Weights()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# %% Paths:

IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels')

# %% Model

model = weights_manager.get_pretrained_model(model_identifier = "Sentinel2_SwinT_SI_RGB",
                                             fpn = True, head = satlaspretrain_models.Head.SEGMENT,
                                             num_categories = 6,
                                             device = device)

model = model.to(device)

# Congelar pesos del backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# %% Criterion

Focal_Loss = True

if Focal_Loss:

    class_weights = calculate_weights_FLoss (MASK_DIR, 6, 512, device)
    criterion = FocalLoss(alpha= class_weights, gamma=2)

else:
    criterion = torch.nn.CrossEntropyLoss()

# %% Train

batch_size = 10
epochs = 1

optimizer = torch.optim.Adam(model.parameters())
TRANSFORM = transforms.ToTensor()

dataset = Dataset(IMAGE_DIR, MASK_DIR, TRANSFORM, device)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
