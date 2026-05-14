"""
Egitim sonuclarini model checkpoint'tan cikartip JSON olarak kaydeder.
Web arayuzu bu JSON'u okuyarak metrikleri gunceller.
"""
import torch
import json
import os
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import WeedDataset, get_val_transforms
from utils import calculate_miou, calculate_precision_recall_f1, calculate_pixel_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# Model yukle
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

checkpoint_path = "model_epoch_25.pth.tar"
if not os.path.exists(checkpoint_path):
    print(f"HATA: {checkpoint_path} bulunamadi!")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Model yuklendi: {checkpoint_path}")

# Validation verisi
VAL_IMG_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"

val_ds = WeedDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=get_val_transforms())
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

# Metrik toplama
all_miou = []
all_pixel_acc = []
all_precisions = [0.0, 0.0, 0.0]
all_recalls = [0.0, 0.0, 0.0]
all_f1 = [0.0, 0.0, 0.0]
batch_count = 0

print("Validation uzerinde metrikler hesaplaniyor...")
with torch.no_grad():
    for images, masks in val_loader:
        images = images.float().to(DEVICE)
        masks = masks.long().to(DEVICE)
        
        preds = model(images)
        
        miou = calculate_miou(preds, masks, NUM_CLASSES)
        pixel_acc = calculate_pixel_accuracy(preds, masks)
        prec, rec, f1 = calculate_precision_recall_f1(preds, masks, NUM_CLASSES)
        
        all_miou.append(miou)
        all_pixel_acc.append(pixel_acc)
        for i in range(3):
            all_precisions[i] += prec[i]
            all_recalls[i] += rec[i]
            all_f1[i] += f1[i]
        batch_count += 1

# Ortalamalar
avg_miou = sum(all_miou) / len(all_miou) if all_miou else 0
avg_pixel_acc = sum(all_pixel_acc) / len(all_pixel_acc) if all_pixel_acc else 0
for i in range(3):
    all_precisions[i] /= max(batch_count, 1)
    all_recalls[i] /= max(batch_count, 1)
    all_f1[i] /= max(batch_count, 1)

avg_f1_total = sum(all_f1) / 3

class_names = ["Arka Plan", "Ekin", "Yabanci Ot"]

print(f"\nmIoU: {avg_miou*100:.2f}%")
print(f"Pixel Accuracy: {avg_pixel_acc*100:.2f}%")
print(f"F1-Score (Ort.): {avg_f1_total*100:.2f}%")

for i, name in enumerate(class_names):
    print(f"  {name}: P={all_precisions[i]*100:.1f}% R={all_recalls[i]*100:.1f}% F1={all_f1[i]*100:.1f}%")

# JSON olarak kaydet (Web arayuzu icin)
metrics = {
    "miou": round(avg_miou * 100, 2),
    "pixel_acc": round(avg_pixel_acc * 100, 2),
    "avg_f1": round(avg_f1_total * 100, 2),
    "classes": []
}

for i, name in enumerate(class_names):
    metrics["classes"].append({
        "name": name,
        "precision": round(all_precisions[i] * 100, 1),
        "recall": round(all_recalls[i] * 100, 1),
        "f1": round(all_f1[i] * 100, 1),
        "iou": round(all_miou[0] * 100, 1) if i == 0 else round(avg_miou * 100, 1)
    })

with open("ui/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrikler 'ui/metrics.json' dosyasina kaydedildi!")
