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
    num_categories = 4,
    device = device_str)

model = model.to(device)

# Congelar pesos del backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# %% Criterion
criterion_type = 'Focal' # Seleccionar entre 'Dice', 'Focal' o 'CE'

if criterion_type == 'Dice':
    criterion = GenDiceLoss(eps = 10, device=device)

elif criterion_type == 'Focal':
    criterion = FocalLoss(alpha= torch.Tensor([1,1,10,1]).to(device), gamma = 1)

elif criterion_type == 'CE':
    criterion = torch.nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = Dataset(IMAGE_DIR, MASK_DIR, transform, device)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 10
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Train
epochs = 10
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    for batch_ix, (images, masks) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images)[0]
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        # calculate batch accuracy
        batch_accuracy = calculate_accuracy(outputs, masks.long())
        # store epoch loss and accuracy
        epoch_loss += loss.item()
        epoch_correct += batch_accuracy * masks.numel()
        epoch_total += masks.numel()       
        # print batch state
        current = (batch_ix + 1) * len(images)
        print(f"loss ({criterion_type}): {loss:.4f}, accuracy: {batch_accuracy:.4f} [{current:>5d}/{len(train_dataset):>5d}]")

    # calculate epoch loss and accuracy
    epoch_loss /= len(train_dataloader)
    epoch_accuracy = epoch_correct / epoch_total
    # print epoch state
    print(f"Epoch {epoch + 1}/{epochs} completed: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

print('Fin modelo actual desarrollado')

# %%
# calculate test accuracy (per classes)
# model in evaluation mode
model.eval()

# initialize variables to store loss and accuracy
test_loss = 0
test_correct = 0
test_total = 0

# variables to store accuracy per class
class_accuracies = {0: 0, 1: 0, 2: 0, 3: 0}  
class_totals = {0: 0, 1: 0, 2: 0, 3: 0}

# deactivate gradient calculation (no convergence, no model 'developing')
with torch.no_grad():
    for batch_ix, (images, masks) in enumerate(test_dataloader):
        outputs = model(images)[0]  # get predictions
        loss = criterion(outputs, masks.long())

        # calculate batch accuracy
        batch_accuracy = calculate_accuracy(outputs, masks.long())

        # calculate class-wise accuracy for class 2
        class_2_accuracy = calculate_class_accuracy(outputs, masks.long(), 2)
        
        # store batch loss and accuracy
        test_loss += loss.item()
        test_correct += batch_accuracy * masks.numel()
        test_total += masks.numel()
        
        # Accumulate class-wise accuracies
        for class_index in range(4):  
            class_accuracies[class_index] += calculate_class_accuracy(outputs, masks.long(), class_index) * (masks == class_index).sum().item()
            class_totals[class_index] += (masks == class_index).sum().item()

        # print state
        current = (batch_ix + 1) * len(images)
        print(f"Test loss: {loss.item():.4f}, Test accuracy: {batch_accuracy:.4f} [{current:>5d}/{len(test_dataset):>5d}], Class 'solar' accuracy: {class_2_accuracy:.4f}")

# calculate mean loss and accuracy
test_loss /= len(test_dataloader)
test_accuracy = test_correct / test_total

# calculate mean accuracy per class
mean_class_accuracies = {class_index: class_accuracies[class_index] / class_totals[class_index] if class_totals[class_index] != 0 else 0 for class_index in class_accuracies}

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print(f"Class 0 Accuracy: {mean_class_accuracies[0]:.4f}")
print(f"Class 1 Accuracy: {mean_class_accuracies[1]:.4f}")
print(f"Class 2 Accuracy: {mean_class_accuracies[2]:.4f}")
print(f"Class 3 Accuracy: {mean_class_accuracies[3]:.4f}")


# %%
