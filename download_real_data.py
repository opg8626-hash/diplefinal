import os
import urllib.request
import zipfile
import shutil
import glob
import random

def download_and_extract(url, extract_to):
    print(f"Veri seti indiriliyor: {url}")
    zip_path = os.path.join(extract_to, "dataset.zip")
    
    # İndirme işlemi
    urllib.request.urlretrieve(url, zip_path)
    print("İndirme tamamlandı. Çıkartılıyor...")
    
    # Zip'ten çıkarma
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(zip_path)
    print("Çıkarma tamamlandı.")

def organize_cwfid_data(base_dir, output_dir, split_ratio=0.8):
    """
    CWFID veri setini train ve val olarak ayırır.
    """
    images_dir = os.path.join(base_dir, "dataset-master", "images")
    masks_dir = os.path.join(base_dir, "dataset-master", "annotations")
    
    # Çıkartılan klasör yapısını kontrol et
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print("Hata: Çıkartılan veri seti klasörleri bulunamadı. Yapı farklı olabilir.")
        return

    # Görselleri bul
    all_images = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    all_images.sort()
    
    # Karıştır
    random.seed(42)
    random.shuffle(all_images)
    
    split_index = int(len(all_images) * split_ratio)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]
    
    def copy_files(image_list, split_name):
        split_img_dir = os.path.join(output_dir, split_name, "images")
        split_mask_dir = os.path.join(output_dir, split_name, "masks")
        
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)
        
        for img_path in image_list:
            base_name = os.path.basename(img_path)
            # Resim ismi 001_image.png, maske ismi 001_annotation.png şeklinde
            prefix = base_name.split("_")[0]
            mask_path = os.path.join(masks_dir, prefix + "_annotation.png")
            
            if not os.path.exists(mask_path):
                # Sadece .png olarak da olabilir (diğer veri setleri için yedek)
                mask_path = os.path.join(masks_dir, os.path.splitext(base_name)[0] + ".png")
            
            if os.path.exists(mask_path):
                shutil.copy(img_path, os.path.join(split_img_dir, base_name))
                shutil.copy(mask_path, os.path.join(split_mask_dir, os.path.basename(mask_path)))
            else:
                print(f"Uyarı: {base_name} için maske bulunamadı.")
                
    print(f"Eğitim (Train) verileri ({len(train_images)} görsel) kopyalanıyor...")
    copy_files(train_images, "train")
    
    print(f"Doğrulama (Val) verileri ({len(val_images)} görsel) kopyalanıyor...")
    copy_files(val_images, "val")
    
    print("Veri seti başarıyla düzenlendi!")

if __name__ == "__main__":
    # CWFID (Crop/Weed Field Image Dataset) github master zip linki
    # Bu veri setinde tarla görüntüleri ve yabancı ot (weed) / ekin (crop) segmentasyon maskeleri bulunur.
    dataset_url = "https://github.com/cwfid/dataset/archive/refs/heads/master.zip"
    
    temp_dir = "temp_dataset"
    output_dir = "data"
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. İndir ve Çıkart
    download_and_extract(dataset_url, temp_dir)
    
    # 2. Düzenle ve Train/Val olarak ayır
    organize_cwfid_data(temp_dir, output_dir)
    
    # 3. Geçici klasörü temizle
    print("Geçici dosyalar temizleniyor...")
    shutil.rmtree(temp_dir)
    
    print(f"İşlem tamam! Gerçek veriler '{output_dir}' klasörüne train ve val olarak eklendi.")
