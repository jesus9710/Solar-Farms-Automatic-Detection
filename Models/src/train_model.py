#%% Libs:

import satlaspretrain_models
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

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
    model_identifier = "Aerial_SwinB_SI",
    fpn = True,
    head = satlaspretrain_models.Head.SEGMENT,
    num_categories = 4,
    device = device_str)

model = model.to(device)

# Congelar pesos del backbone
for param in model.parameters():
    param.requires_grad = False

for param in model.upsample.parameters():
    param.requires_grad = True

for param in model.head.parameters():
    param.requires_grad = True

# %% Criterion
criterion_type = 'CE' # Seleccionar entre 'Dice', 'Focal' o 'CE'

if criterion_type == 'Dice':
    #criterion = GenDiceLoss(eps = 10, device=device)
    criterion = smp.losses.DiceLoss(mode='multiclass', classes=4, eps=1e-07)

elif criterion_type == 'Focal':
    criterion = FocalLoss(alpha= torch.Tensor([1,1,100,1]).to(device), gamma = 1)

elif criterion_type == 'CE':
    criterion = torch.nn.CrossEntropyLoss()

transform = transforms.ToTensor()

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 40
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Train
epochs = 30
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    for batch_ix, (images, masks) in enumerate(train_dataloader):
        optimizer.zero_grad()
        out_backbone = model.backbone(images)
        out_fpn = model.fpn(out_backbone)
        out_upsample = model.upsample(out_fpn)
        outputs, loss = model.head(images, out_upsample, masks)
        #loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), (batch_ix + 1) * len(images)
        print(f"loss ({criterion_type}): {loss:.4f} [{current:>5d}/{len(train_dataset):>5d}]")

print('Fin modelo actual desarrollado ')

# %% Prediction

model.eval()

soft_preds = []
target = []
images_list = []

with torch.no_grad():
    for image, mask in train_dataloader:
        out_backbone = model.backbone(images)
        out_fpn = model.fpn(out_backbone)
        out_upsample = model.upsample(out_fpn)
        outputs, _ = model.head(images, out_upsample, masks)
        soft_preds.append(outputs)
        target.append(mask)
        images_list.append(image)

soft_preds = torch.cat(soft_preds, dim=0)
target = torch.cat(target, dim=0)
images_ = torch.cat(images_list, dim=0)

# %% Hard output

hard_pred = torch.argmax(soft_preds, dim=1)
print(hard_pred.min())

# %% Base Line

base_preds = (torch.ones(*hard_pred.shape) * 3).long().to(device)

# %% Evaluation

tp, fp, fn, tn = smp.metrics.get_stats(base_preds, target, mode='multiclass',num_classes=4)
base_score = smp.metrics.functional.accuracy(tp[:,2], fp[:,2], fn[:,2], tn[:,2]).mean()

tp, fp, fn, tn = smp.metrics.get_stats(hard_pred, target, mode='multiclass',num_classes=4)
score = smp.metrics.functional.accuracy(tp[:,2], fp[:,2], fn[:,2], tn[:,2]).mean()

print(f'BaseLine score: {base_score}')
print(f'model score: {score}')
