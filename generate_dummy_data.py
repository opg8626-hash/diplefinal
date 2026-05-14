import os
import cv2
import numpy as np

def create_dummy_data(base_dir, num_samples=10):
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Rastgele 256x256 RGB görüntü oluştur (gürültü benzeri)
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Rastgele 256x256 maske oluştur (sınıflar: 0, 1, 2)
        mask = np.random.randint(0, 3, (256, 256), dtype=np.uint8)
        
        # Görüntüleri .jpg, maskeleri .png olarak kaydet
        img_name = f"dummy_{i}.jpg"
        mask_name = f"dummy_{i}.png"
        
        cv2.imwrite(os.path.join(images_dir, img_name), image)
        cv2.imwrite(os.path.join(masks_dir, mask_name), mask)

if __name__ == "__main__":
    print("Test için dummy (rastgele) veri seti oluşturuluyor...")
    create_dummy_data("data/train", num_samples=20)
    create_dummy_data("data/val", num_samples=5)
    print("Dummy veri seti oluşturuldu! Artık 'python train.py' çalıştırabilirsiniz.")
