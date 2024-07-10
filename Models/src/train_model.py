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
criterion_type = 'Dice' # Seleccionar entre 'Dice', 'Focal' o 'CE'

if criterion_type == 'Dice':
    criterion = GenDiceLoss(eps = 10, device=device)

elif criterion_type == 'Focal':
    criterion = FocalLoss(alpha= torch.Tensor([1,10,1,1]).to(device), gamma = 1)

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
import torch
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

epochs = 10  # Cambié a 10 para ver mejor las tendencias
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()  # Ejemplo, reemplaza con la función de pérdida que estés usando

# Variables para almacenar métricas
iou_scores = []
f1_scores = []
f2_scores = []
accuracies = []
recalls = []
roc_aucs = []
losses = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()

    # Reset metrics at the beginning of each epoch
    iou_scores_epoch = []
    f1_scores_epoch = []
    f2_scores_epoch = []
    accuracies_epoch = []
    recalls_epoch = []
    roc_aucs_epoch = []
    losses_epoch = []

    for batch_ix, (images, masks) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images)[0]
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        loss_value, current = loss.item(), (batch_ix + 1) * len(images)

        # Calculate metrics
        tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks.long(), mode='multilabel', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        # Calculate ROC AUC for each class and take the average
        roc_auc = roc_auc_score(masks.cpu().numpy().reshape(-1), outputs.sigmoid().cpu().detach().numpy().reshape(-1))
        roc_aucs_epoch.append(roc_auc)

        # Append metrics for current batch
        losses_epoch.append(loss_value)
        iou_scores_epoch.append(iou_score.item())
        f1_scores_epoch.append(f1_score.item())
        f2_scores_epoch.append(f2_score.item())
        accuracies_epoch.append(accuracy.item())
        recalls_epoch.append(recall.item())

        print(f"loss ({criterion}): {loss_value:.4f} [{current:>5d}/{len(train_dataset):>5d}]")

    # Calculate average metrics for the epoch
    losses.append(sum(losses_epoch) / len(losses_epoch))
    iou_scores.append(sum(iou_scores_epoch) / len(iou_scores_epoch))
    f1_scores.append(sum(f1_scores_epoch) / len(f1_scores_epoch))
    f2_scores.append(sum(f2_scores_epoch) / len(f2_scores_epoch))
    accuracies.append(sum(accuracies_epoch) / len(accuracies_epoch))
    recalls.append(sum(recalls_epoch) / len(recalls_epoch))
    roc_aucs.append(sum(roc_aucs_epoch) / len(roc_aucs_epoch))

    print(f"Epoch {epoch + 1} metrics: IoU: {iou_scores[-1]:.4f}, F1: {f1_scores[-1]:.4f}, "
          f"F2: {f2_scores[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}, Recall: {recalls[-1]:.4f}, "
          f"ROC AUC: {roc_aucs[-1]:.4f}")

print('Fin modelo actual desarrollado')

# Plotting the metrics
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(18, 6))

# Loss Plot
plt.subplot(1, 3, 1)
plt.plot(epochs_range, losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# ROC AUC Plot
plt.subplot(1, 3, 2)
plt.plot(epochs_range, roc_aucs, label='ROC AUC', color='orange')
plt.xlabel('Epochs')
plt.ylabel('ROC AUC')
plt.title('ROC AUC over Epochs')
plt.legend()

# Accuracy Plot
plt.subplot(1, 3, 3)
plt.plot(epochs_range, accuracies, label='Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
# %%
