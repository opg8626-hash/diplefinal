import os
import cv2
import numpy as np
import glob
import albumentations as A
import random

dataset_dir = "weedsgalore_extracted/weedsgalore-dataset"
output_img_dir = "data/train/images"
output_mask_dir = "data/train/masks"

# Orijinal CWFID resimlerini silip temiz bir sayfa açıyoruz
import shutil
shutil.rmtree(output_img_dir, ignore_errors=True)
shutil.rmtree(output_mask_dir, ignore_errors=True)
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# 1. WeedsGalore verilerini topla ve dönüştür
date_folders = [f.path for f in os.scandir(dataset_dir) if f.is_dir() and "2023" in f.name]

base_names = []
for folder in date_folders:
    mask_files = glob.glob(os.path.join(folder, "semantics", "*.png"))
    for mf in mask_files:
        name = os.path.basename(mf).replace(".png", "")
        # Sadece R, G, B dosyalarının varlığından emin ol
        if os.path.exists(os.path.join(folder, "images", f"{name}_R.png")):
            base_names.append((folder, name))

print(f"Toplam WeedsGalore benzersiz fotoğrafı: {len(base_names)}")

saved_count = 0
generated_images = []
generated_masks = []

for folder, name in base_names:
    r_path = os.path.join(folder, "images", f"{name}_R.png")
    g_path = os.path.join(folder, "images", f"{name}_G.png")
    b_path = os.path.join(folder, "images", f"{name}_B.png")
    mask_path = os.path.join(folder, "semantics", f"{name}.png")
    
    # R, G, B kanallarını gri tonlamalı oku (çünkü ayrı kaydedilmişler)
    r = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    
    # RGB Görüntüyü birleştir
    rgb = cv2.merge([b, g, r]) # OpenCV BGR formatı kullanır
    
    # Maskeyi oku
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Maske Değerlerini Güncelle (0: Arka Plan, 1: Ekin, >1: Yabancı Ot)
    new_mask = np.zeros_like(mask)
    new_mask[mask == 1] = 1 # Ekin
    new_mask[mask > 1] = 2  # Yabancı Ot (WeedsGalore'da 3, 4, 5 gibi farklı ot tipleri var)
    
    # Yeniden Boyutlandır (Hızlı eğitim için 256x256)
    rgb = cv2.resize(rgb, (256, 256))
    new_mask = cv2.resize(new_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Orijinali kaydet
    cv2.imwrite(os.path.join(output_img_dir, f"{name}.jpg"), rgb)
    cv2.imwrite(os.path.join(output_mask_dir, f"{name}.png"), new_mask)
    
    generated_images.append(rgb)
    generated_masks.append(new_mask)
    saved_count += 1

print(f"{saved_count} gerçek görüntü çıkarıldı.")

# 2. 1000'e tamamlamak için Data Augmentation (Veri Çoğaltma)
target_count = 1000
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
    A.GaussNoise(p=0.3)
])

print(f"{target_count - saved_count} adet ekstra görüntü Augmentation ile üretiliyor...")

while saved_count < target_count:
    # Rastgele bir orijinal görüntü seç
    idx = random.randint(0, len(generated_images) - 1)
    img = generated_images[idx]
    mask = generated_masks[idx]
    
    # Augmentation uygula
    augmented = augmentor(image=img, mask=mask)
    aug_img = augmented['image']
    aug_mask = augmented['mask']
    
    # Kaydet
    new_name = f"aug_{saved_count}.jpg"
    new_mask_name = f"aug_{saved_count}.png"
    cv2.imwrite(os.path.join(output_img_dir, new_name), aug_img)
    cv2.imwrite(os.path.join(output_mask_dir, new_mask_name), aug_mask)
    
    saved_count += 1

print(f"Toplam {saved_count} adet görüntü ve maske başarıyla 'data/train' klasörüne kaydedildi!")
