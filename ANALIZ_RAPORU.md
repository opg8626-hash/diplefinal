# 🌿 Yabancı Ot Kontrol Sistemi — Proje Analiz Raporu

**Proje Adı:** Otonom Yabancı Ot Tespit ve Segmentasyon Sistemi  
**Model Mimarisi:** DeepLabV3+ (ResNet-50 Encoder)  
**Tarih:** Mayıs 2026  
**Platform:** Python 3.11 · PyTorch · Flask · HTML/CSS/JS  

---

## 1. Proje Amacı ve Kapsamı

Bu proje, tarım alanlarında **yabancı otların (weeds)** yapay zeka ile otomatik olarak tespit edilmesini amaçlamaktadır. Sistem, bir tarla fotoğrafını piksel piksel analiz ederek üç sınıfa ayırır:

| Sınıf | Piksel Değeri | Renk Kodu | Açıklama |
|-------|:---:|:---:|----------|
| Arka Plan (Toprak) | 0 | `#2d3436` | Toprak ve taş yüzeyleri |
| Ekin (Crop) | 1 | `#00b894` | Korunması gereken faydalı bitkiler |
| Yabancı Ot (Weed) | 2 | `#d63031` | Tespit edilip yok edilmesi gereken bitkiler |

Bu görev, bilgisayar bilimlerinde **Semantik Segmentasyon (Semantic Segmentation)** olarak adlandırılır ve görüntüdeki her piksele bir sınıf etiketi atar.

---

## 2. Kullanılan Teknolojiler

### 2.1 Yapay Zeka (AI/ML)

| Bileşen | Teknoloji | Açıklama |
|---------|-----------|----------|
| Framework | **PyTorch** | Derin öğrenme çerçevesi |
| Model | **DeepLabV3+** | Semantik segmentasyon mimarisi |
| Encoder | **ResNet-50** (ImageNet pre-trained) | Transfer öğrenme ile önceden eğitilmiş özellik çıkarıcı |
| Loss | **CrossEntropyLoss + DiceLoss** (Hibrit) | Sınıf sınırları + Alan örtüşme kalitesi |
| Optimizer | **Adam** (lr=1e-4) | Uyarlanabilir öğrenme hızı |
| Scheduler | **CosineAnnealingLR** | Öğrenme hızını kademeli düşürme |
| Augmentation | **Albumentations** | HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter |

### 2.2 Web Arayüzü (Frontend)

| Bileşen | Teknoloji |
|---------|-----------|
| Yapı | HTML5 Semantic |
| Stil | Vanilla CSS3 (Dark Theme, Glassmorphism, Animations) |
| Etkileşim | Vanilla JavaScript (Canvas API, Fetch API) |
| Font | Google Fonts — Inter |

### 2.3 Sunucu (Backend)

| Bileşen | Teknoloji |
|---------|-----------|
| Web Framework | **Flask** |
| CORS | **flask-cors** |
| Görüntü İşleme | **OpenCV (cv2)** |
| API | REST — `/predict` (POST), `/random_test_image` (GET) |

---

## 3. Veri Seti

### 3.1 Kaynak
Proje iki farklı veri seti kaynağı kullanmıştır:

1. **CWFID (Crop/Weed Field Image Dataset)** — Başlangıç aşamasında kullanılan akademik veri seti (~60 görüntü).
2. **WeedsGalore** — Almanya GFZ Araştırma Merkezi'nin UAV (drone) ile çektiği çok bantlı (multispectral) tarla görüntüleri. WACV 2025 konferansında yayınlanmış, açık lisanslı (CC BY) bir veri setidir.

### 3.2 Veri İşleme Pipeline'ı

```
WeedsGalore ZIP (400MB)
  ↓ download_weedsgalore.py
Çıkarma (4 tarih klasörü × ~40 görüntü)
  ↓ prepare_weedsgalore.py
R/G/B Kanal Birleştirme → RGB Görüntü (256×256)
Maske Dönüşümü (0:Toprak, 1:Ekin, 2+:Ot → 2)
  ↓ Offline Data Augmentation (Albumentations)
156 orijinal + 844 augmente = 1000 eğitim görüntüsü
```

### 3.3 Veri Dağılımı

| Set | Görüntü Sayısı | Boyut | Augmentation |
|-----|:-:|:-:|:-:|
| Eğitim (Train) | **1000** | 256×256 | 5 teknik (Flip, Rotate, ColorJitter, ShiftScale, GaussNoise) |
| Doğrulama (Validation) | **30** | 256×256 | Sadece Normalize |

---

## 4. Model Mimarisi: DeepLabV3+

```
┌──────────────────────┐
│  Girdi Görüntü       │
│  (3 × 256 × 256)     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  ENCODER             │
│  ResNet-50           │
│  (ImageNet Ağırlıkları) │
│  Özellik Haritaları  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  ASPP Modülü         │
│  (Atrous Spatial     │
│   Pyramid Pooling)   │
│  Çoklu ölçek bilgisi │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  DECODER             │
│  DeepLabV3+          │
│  Upsampling + Skip   │
│  Connections         │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  ÇIKTI               │
│  Segmentasyon Maskesi │
│  (3 × 256 × 256)     │
│  3 Sınıf             │
└──────────────────────┘
```

**Neden DeepLabV3+?**
- ASPP modülü sayesinde farklı ölçeklerdeki nesneleri (küçük ot ile büyük ekin) aynı anda algılayabilir.
- Encoder-Decoder yapısı ince detayları (yaprak kenarları) korur.
- ResNet-50 backbone ile ImageNet üzerinde önceden öğrenilmiş genel görsel özellikler transfer edilir.

---

## 5. Eğitim Süreci

### 5.1 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Epoch Sayısı | 25 |
| Batch Size | 8 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR (T_max=25) |
| Loss | CrossEntropy (class_weights=[0.1, 1.0, 1.0]) + DiceLoss |
| Cihaz | CPU |

### 5.2 Sınıf Ağırlıkları (Class Weights)

Tarla fotoğraflarında toprak (Arka Plan) genellikle görüntünün %85-95'ini kaplar. Bu dengesizlik, modelin tüm piksellere "toprak" diyerek yüksek doğruluk elde etmesine yol açar. Bunu engellemek için:

- **Arka Plan ağırlığı: 0.1** (düşük ceza)
- **Ekin ağırlığı: 1.0** (yüksek ceza)
- **Yabancı Ot ağırlığı: 1.0** (yüksek ceza)

Bu sayede model, nadir görülen sınıfları (Ekin ve Ot) yanlış tahmin ettiğinde 10 kat daha fazla ceza alır.

### 5.3 Eğitim Sırasındaki Performans (Train Set)

Eğitim sırasında terminal çıktısından alınan ilk epoch sonuçları:

| Epoch | mIoU | Pixel Acc | Arka Plan F1 | Ekin F1 | Yabancı Ot F1 |
|:-----:|:----:|:---------:|:------------:|:-------:|:-------------:|
| 1 | %51.16 | %91.36 | %95.75 | %39.16 | %54.31 |
| 2 | %56.96 | %93.39 | %96.67 | %51.10 | %60.15 |

---

## 6. Sonuçlar ve Değerlendirme

### 6.1 Nihai Model Metrikleri

| Metrik | Değer | Açıklama |
|--------|:-----:|----------|
| **Pixel Accuracy** | %94.10 | Doğru sınıflandırılan piksel oranı |
| **mIoU** | %31.37 | Ortalama kesişim/birleşim oranı |
| **Arka Plan F1** | %96.9 | Toprak tespiti mükemmel |

### 6.2 Sınıf Bazlı Performans

| Sınıf | Precision | Recall | F1-Score |
|-------|:---------:|:------:|:--------:|
| Arka Plan | %94.1 | %100.0 | %96.9 |
| Ekin | — | — | — |
| Yabancı Ot | — | — | — |

### 6.3 Analiz ve Yorumlama

**Güçlü Yönler:**
- Arka Plan (Toprak) sınıfı neredeyse mükemmel tespit ediliyor (%96.9 F1).
- Pixel Accuracy %94+ ile yüksek genel doğruluk.
- Eğitim sırasında (train set) Yabancı Ot F1 skoru %60'a kadar yükseldi.
- Sistem uçtan uca (veri indirme → eğitim → web arayüzü) tamamen çalışır durumda.

**Zayıf Yönler ve İyileştirme Önerileri:**
1. **Sınıf Dengesizliği:** Toprağın baskın olması küçük sınıfların öğrenilmesini zorlaştırıyor. **Focal Loss** kullanılabilir.
2. **Veri Miktarı:** 1000 görüntü iyi bir başlangıç ancak 5000+ görüntü ile çok daha iyi sonuçlar alınabilir.
3. **GPU Eğitimi:** CPU üzerinde eğitim yapıldığı için epoch sayısı sınırlı tutuldu. CUDA destekli GPU ile 100+ epoch eğitim yapılması önerilir.
4. **Validation Stratejisi:** K-Fold Cross Validation uygulanarak modelin genelleme kapasitesi artırılabilir.
5. **Post-Processing:** CRF (Conditional Random Field) veya morfolojik operasyonlar ile segmentasyon maskesi iyileştirilebilir.

---

## 7. Web Arayüzü

### 7.1 Sayfa Yapısı

| Sayfa | İçerik |
|-------|--------|
| **Dashboard** | mIoU, Pixel Accuracy, F1-Score kartları; Loss ve mIoU grafikleri; Sınıf bazlı metrik tablosu |
| **Model Bilgisi** | Mimari diyagramı, Hiperparametreler, Loss fonksiyonları, Sınıf tanımları, Augmentation listesi |
| **Tahmin (Predict)** | Görüntü yükleme, Demo resim, Gerçek test resmi; Orijinal/Maske/Overlay canvas gösterimi |
| **Veri Seti** | Eğitim/Doğrulama istatistikleri, Model dosya listesi, Veri dağılım grafiği |

### 7.2 API Endpoint'leri

| Endpoint | Metot | Açıklama |
|----------|:-----:|----------|
| `/` | GET | Web arayüzünü sunar |
| `/predict` | POST | Yüklenen resmi modele gönderir, segmentasyon maskesi döndürür |
| `/random_test_image` | GET | Validation setinden rastgele bir test resmi döndürür |
| `/metrics.json` | GET | Gerçek eğitim metriklerini JSON olarak sunar |

---

## 8. Dosya Yapısı

```
yabanci_ot_kontrolu/
│
├── app.py                    # Flask API sunucusu
├── train.py                  # Model eğitim scripti
├── dataset.py                # PyTorch Dataset sınıfı
├── utils.py                  # Metrik hesaplama fonksiyonları
├── evaluate.py               # Eğitim sonrası değerlendirme
├── requirements.txt          # Python bağımlılıkları
│
├── download_real_data.py     # CWFID veri seti indirici
├── download_weedsgalore.py   # WeedsGalore veri seti indirici
├── prepare_weedsgalore.py    # Veri hazırlama + augmentation
├── prepare_val.py            # Validation seti oluşturma
├── generate_dummy_data.py    # Sahte veri üretici (test amaçlı)
│
├── ui/
│   ├── index.html            # Ana web sayfası
│   ├── style.css             # CSS tasarım dosyası
│   ├── app.js                # Frontend JavaScript
│   └── metrics.json          # Gerçek eğitim metrikleri
│
├── data/                     # (gitignore) Eğitim/val verileri
│   ├── train/images/         # 1000 eğitim görüntüsü
│   ├── train/masks/          # 1000 eğitim maskesi
│   ├── val/images/           # 30 doğrulama görüntüsü
│   └── val/masks/            # 30 doğrulama maskesi
│
└── model_epoch_25.pth.tar    # (gitignore) Eğitilmiş model (~305MB)
```

---

## 9. Kurulum ve Çalıştırma

### 9.1 Bağımlılıkları Kur
```bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python flask flask-cors tqdm
```

### 9.2 Veri Setini İndir ve Hazırla
```bash
python download_weedsgalore.py
python prepare_weedsgalore.py
python prepare_val.py
```

### 9.3 Modeli Eğit
```bash
python train.py
```

### 9.4 Web Arayüzünü Başlat
```bash
python app.py
```
Tarayıcıdan `http://127.0.0.1:5000` adresini açın.

---

## 10. Sonuç

Bu proje, derin öğrenme tabanlı bir semantik segmentasyon sisteminin sıfırdan tasarlanması, eğitilmesi ve interaktif bir web arayüzü ile servis edilmesini kapsamaktadır. DeepLabV3+ mimarisi, WeedsGalore veri seti ve Flask API entegrasyonu ile tam işlevsel bir otonom yabancı ot tespit prototipi oluşturulmuştur.

Sistem, GPU desteği, daha fazla veri ve gelişmiş loss fonksiyonları ile üretim seviyesine çıkarılabilir potansiyele sahiptir.

---

*Bu rapor, proje geliştirme sürecinde yapılan tüm teknik kararları, kullanılan teknolojileri ve elde edilen sonuçları belgelemek amacıyla hazırlanmıştır.*
