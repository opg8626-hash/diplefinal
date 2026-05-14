import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WeedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Veri seti yükleyici sınıfımız.
        image_dir: Görüntülerin bulunduğu klasör
        mask_dir: Maskelerin bulunduğu klasör (Maskeler 0, 1, 2 gibi sınıf indeksleri içermelidir)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Maske isimlerinin görüntülerle aynı olduğunu varsayıyoruz. 
        # (Ör: img1.jpg için img1.png gibi bir maske olabilir, projeye göre revize edilebilir)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Maskeyi gri tonlamalı okuyoruz. Her pikselin değeri bir sınıfı temsil etmeli (0: Toprak, 1: Ekin, 2: Ot)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            unique_vals = np.unique(mask)
            # Eski CWFID formatı (0, 76, 149) ise dönüştür
            if 76 in unique_vals or 149 in unique_vals:
                new_mask = np.zeros_like(mask)
                new_mask[mask == 76] = 1   # Ekin
                new_mask[mask == 149] = 2  # Yabancı Ot
                mask = new_mask
            else:
                # WeedsGalore formatı (0, 1, 2) — olduğu gibi kullan, >2 olanları 2'ye sınırla
                mask = np.clip(mask, 0, 2)
        else:
            # Maske yoksa hata vermemesi için boş maske oluştur (Test aşamasında silinmelidir)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask.long()

def get_train_transforms():
    """
    Eğitim (Train) verisi için Data Augmentation (Veri Çoğaltma) işlemleri.
    Dökümantasyonda bahsedilen çevirme, renk değişimleri gibi teknikleri içerir.
    """
    return A.Compose([
        A.Resize(height=256, width=256), # Boyutlandırma
        A.HorizontalFlip(p=0.5), # Yatay çevirme
        A.VerticalFlip(p=0.5), # Dikey çevirme
        A.RandomRotate90(p=0.5), # Rastgele 90 derece çevirme
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), # Işık ve renk manipülasyonu
        A.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet ortalamaları
            std=[0.229, 0.224, 0.225],  # ImageNet standart sapmaları
        ), 
        ToTensorV2(), # PyTorch Tensor formatına dönüştürme
    ])

def get_val_transforms():
    """
    Doğrulama (Validation) verisi için dönüşümler. Sadece boyutlandırma ve normalize edilir.
    """
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
