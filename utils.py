import torch
import numpy as np

def calculate_miou(preds, labels, num_classes):
    """
    mIoU (mean Intersection over Union) metriği hesaplar.
    Dökümantasyonda belirtildiği gibi segmentasyonun en önemli başarı kriteridir.
    """
    preds = torch.argmax(preds, dim=1)
    ious = []
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return sum(valid_ious) / len(valid_ious) if valid_ious else 0.0


def calculate_precision_recall_f1(preds, labels, num_classes):
    """
    Her sınıf için Precision, Recall ve F1-Score hesaplar.
    
    Precision = TP / (TP + FP) → "Ot dediğimiz şey gerçekten ot mu?"
    Recall    = TP / (TP + FN) → "Tarladaki tüm otları bulabildik mi?"
    F1-Score  = 2 * (Precision * Recall) / (Precision + Recall) → Harmonik ortalama
    """
    preds = torch.argmax(preds, dim=1)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        
        tp = (pred_inds & target_inds).sum().item()  # True Positive
        fp = (pred_inds & ~target_inds).sum().item() # False Positive
        fn = (~pred_inds & target_inds).sum().item() # False Negative
        
        # Precision hesabı
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall hesabı
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score hesabı (Harmonik ortalama)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return precisions, recalls, f1_scores


def calculate_pixel_accuracy(preds, labels):
    """
    Piksel Doğruluğu (Pixel Accuracy) hesaplar.
    Toplam doğru tahmin edilen piksel sayısı / Toplam piksel sayısı
    """
    preds = torch.argmax(preds, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total


def print_metrics_table(precisions, recalls, f1_scores, miou, pixel_acc, class_names=None):
    """
    Tüm metrikleri güzel bir tablo formatında terminale yazdırır.
    """
    if class_names is None:
        class_names = ["Arka Plan", "Ekin", "Yabanci Ot"]
    
    print("\n" + "=" * 65)
    print(f"{'SINIF':<15} {'PRECISION':>12} {'RECALL':>12} {'F1-SCORE':>12}")
    print("-" * 65)
    
    for i, name in enumerate(class_names):
        print(f"  {name:<13} {precisions[i]*100:>10.2f}%  {recalls[i]*100:>10.2f}%  {f1_scores[i]*100:>10.2f}%")
    
    print("-" * 65)
    
    # Ortalamalar (Macro Average)
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    print(f"  {'ORTALAMA':<13} {avg_p*100:>10.2f}%  {avg_r*100:>10.2f}%  {avg_f1*100:>10.2f}%")
    print("=" * 65)
    print(f"  Pixel Accuracy : {pixel_acc*100:.2f}%")
    print(f"  mIoU           : {miou*100:.2f}%")
    print("=" * 65 + "\n")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Model kaydediliyor...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Model yukleniyor...")
    model.load_state_dict(checkpoint["state_dict"])
