import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import WeedDataset, get_train_transforms, get_val_transforms
from utils import calculate_miou, calculate_precision_recall_f1, calculate_pixel_accuracy, print_metrics_table, save_checkpoint
from tqdm import tqdm
import os

# --- HİPERPARAMETRELER ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
NUM_CLASSES = 3 # 0: Arka Plan, 1: Ekin, 2: Yabancı Ot

# Veri Yolları (Kendi verinize göre güncelleyiniz)
TRAIN_IMG_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"
VAL_IMG_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"

def train_fn(loader, model, optimizer, ce_loss_fn, dice_loss_fn, scaler):
    """
    Tek bir epoch için eğitim fonksiyonu.
    """
    loop = tqdm(loader, leave=True)
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # İleri yayılım (Mixed Precision kullanılarak)
        with torch.cuda.amp.autocast():
            predictions = model(images)
            
            # Loss Hesabı: CrossEntropy + Dice Loss (Dökümantasyonda önerilen hibrit yapı)
            loss_ce = ce_loss_fn(predictions, masks)
            loss_dice = dice_loss_fn(predictions, masks)
            loss = loss_ce + loss_dice

        # Geri yayılım ve Optimizasyon
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        # TQDM üzerinde anlık loss gösterimi
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def check_accuracy(loader, model, device="cuda"):
    """
    Dogrulama (Validation) verisi uzerinde tum metrikleri hesaplar:
    - Pixel Accuracy
    - mIoU
    - Precision, Recall, F1-Score (sinif bazli + ortalama)
    """
    all_preds = []
    all_masks = []
    model.eval()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            all_preds.append(preds)
            all_masks.append(masks)

    # Tum batch'leri birlestir
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Metrikleri hesapla
    pixel_acc = calculate_pixel_accuracy(all_preds, all_masks)
    miou = calculate_miou(all_preds, all_masks, NUM_CLASSES)
    precisions, recalls, f1_scores = calculate_precision_recall_f1(all_preds, all_masks, NUM_CLASSES)

    # Tabloyu yazdir
    print_metrics_table(precisions, recalls, f1_scores, miou, pixel_acc)

    model.train()
    return miou

def main():
    print(f"Çalışılan Cihaz: {DEVICE}")

    # Gerekli veri klasörlerinin var olup olmadığını kontrol et
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_MASK_DIR, exist_ok=True)

    # 1. Veri Yükleyicileri (DataLoaders)
    train_ds = WeedDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=get_train_transforms())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)

    val_ds = WeedDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

    # 2. Model (Pre-trained DeepLabV3+ with ResNet50)
    # YOLO dışında yüksek doğruluklu PyTorch alternatifi
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet", # Transfer Learning
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    # 3. Loss Fonksiyonları (BCE/CrossEntropy + Dice Loss Hibriti)
    # Sınıf Ağırlıkları (Arka plan çok fazla olduğu için ot ve ekine daha fazla önem ver)
    class_weights = torch.tensor([0.1, 1.0, 1.0]).to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = smp.losses.DiceLoss(mode="multiclass")

    # 4. Optimizasyon
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision (Eğitim hızı için)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. Eğitim Döngüsü
    print("Eğitime başlanıyor...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] ---")
        
        # Eğit
        train_loss = train_fn(train_loader, model, optimizer, ce_loss_fn, dice_loss_fn, scaler)
        
        # Learning Rate Güncelle
        scheduler.step()
        
        # Doğrula ve Değerlendir
        miou = check_accuracy(val_loader, model, device=DEVICE)

        # Modeli kaydet
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"model_epoch_{epoch+1}.pth.tar")

if __name__ == "__main__":
    main()
