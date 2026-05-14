import os
import cv2
import numpy as np
import torch
import base64
import glob
import random
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__, static_folder='ui')
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# Modelin Yüklenmesi
print("Model yükleniyor...")
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

checkpoint_path = "model_epoch_25.pth.tar"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Model başarıyla yüklendi: {checkpoint_path}")
else:
    print(f"Uyarı: {checkpoint_path} bulunamadı!")

# Görüntü işleme kuralları (Validation ile aynı)
transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])

@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('ui', path)

@app.route('/random_test_image')
def random_test_image():
    images = glob.glob('data/val/images/*.*')
    if not images:
        return jsonify({"error": "Test resmi bulunamadı"}), 404
    img_path = random.choice(images)
    return send_file(img_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
        
    file = request.files['file']
    
    # Gelen dosyayı OpenCV formatında oku
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transformatörden geçir ve modele ver
    augmented = transform(image=image)
    tensor_img = augmented["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        preds = model(tensor_img)
        preds = torch.softmax(preds, dim=1)
        preds_class = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
        
    # Sınıf Renkleri (Frontend JS ile uyumlu): 
    # Arka Plan: [45, 52, 54] (Koyu Gri)
    # Ekin: [0, 184, 148] (Yeşil)
    # Ot: [214, 48, 49] (Kırmızı)
    class_colors = [
        [45, 52, 54],
        [0, 184, 148],
        [214, 48, 49]
    ]
    
    # Çıktı maskesini renkli RGBA formata çevir
    mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    bg_count, crop_count, weed_count = 0, 0, 0
    total_pixels = 256 * 256
    
    for c_idx, color in enumerate(class_colors):
        mask_idx = (preds_class == c_idx)
        if c_idx == 0: bg_count = np.sum(mask_idx)
        elif c_idx == 1: crop_count = np.sum(mask_idx)
        elif c_idx == 2: weed_count = np.sum(mask_idx)
        
        mask_rgba[mask_idx] = color + [255] # R, G, B, Alpha(255)
        
    # Maskeyi Frontend'e göndermek için Base64 yap
    # OpenCV'de encode yaparken BGR(A) bekler, biz RGBA oluşturduk, dönüştürüyoruz:
    _, buffer = cv2.imencode('.png', cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGRA))
    mask_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "mask_base64": f"data:image/png;base64,{mask_b64}",
        "bgPct": round(bg_count / total_pixels * 100, 1),
        "cropPct": round(crop_count / total_pixels * 100, 1),
        "weedPct": round(weed_count / total_pixels * 100, 1)
    })

if __name__ == '__main__':
    print("Web sunucusu başlatılıyor: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
