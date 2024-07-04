#%% Libs:

import satlaspretrain_models
import torch
import torchvision.transforms as transforms

from pathlib import Path

from utils import *

weights_manager = satlaspretrain_models.Weights()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"

# %% Paths:
IMAGE_DIR = Path.cwd().parent.joinpath('data/features')
MASK_DIR = Path.cwd().parent.joinpath('data/labels')

# %% Model
model = weights_manager.get_pretrained_model(
    model_identifier="Sentinel2_SwinT_SI_RGB",
    fpn=True, head=satlaspretrain_models.Head.SEGMENT, 
    num_categories=6,
    device=device_str)

model = model.to(device)

# Congelar pesos del backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# %% Criterion
Focal_Loss = False

if Focal_Loss:
    class_weights = calculate_weights_FLoss(MASK_DIR, 6, 512, device)
    criterion = FocalLoss(alpha=class_weights, gamma=2)
else:
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
        if (batch_ix + 1) % 10 == 0:
            print(f"Batch {batch_ix + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

# # %% Evaluate
# model.eval()
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for images, masks in test_dataloader:
#         outputs = model(images)[0]
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.append(preds.cpu().numpy().flatten())
#         all_labels.append(masks.cpu().numpy().flatten())

# all_preds = np.concatenate(all_preds)
# all_labels = np.concatenate(all_labels)

# # %% Confusion Matrix
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5])
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2, 3, 4, 5])
# disp.plot(cmap=plt.cm.Blues)
# plt.show()

print('Fin modelo actual desarrollado ')