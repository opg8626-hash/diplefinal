"""
WeedsGalore verisinden validation seti olusturur.
data/train icerisinden rastgele 100 resmi data/val klasorune tasir.
"""
import os
import shutil
import random

train_img = "data/train/images"
train_mask = "data/train/masks"
val_img = "data/val/images"
val_mask = "data/val/masks"

# Eski val verisini temizle
shutil.rmtree(val_img, ignore_errors=True)
shutil.rmtree(val_mask, ignore_errors=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(val_mask, exist_ok=True)

# Train icerisindeki orijinal (augment olmayan) resimleri sec
all_images = [f for f in os.listdir(train_img) if not f.startswith("aug_")]
print(f"Orijinal goruntu sayisi: {len(all_images)}")

# Rastgele 30 tanesini val'e tasi
random.seed(42)
val_samples = random.sample(all_images, min(30, len(all_images)))

for img_name in val_samples:
    mask_name = img_name.replace(".jpg", ".png")
    
    # Kopyala (tasima degil, train'de de kalsin)
    src_img = os.path.join(train_img, img_name)
    src_mask = os.path.join(train_mask, mask_name)
    
    if os.path.exists(src_img) and os.path.exists(src_mask):
        shutil.copy2(src_img, os.path.join(val_img, img_name))
        shutil.copy2(src_mask, os.path.join(val_mask, mask_name))

print(f"Validation setine {len(os.listdir(val_img))} goruntu kopyalandi.")
