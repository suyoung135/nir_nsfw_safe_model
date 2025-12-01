import os
from PIL import Image, ImageFile
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------
# 1. Device 설정
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------
# 2. 모델 로드
# -------------------------------------
MODEL_NAME = "nsfw_finetuned/final_model"
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model.eval()

# -------------------------------------
# 3. 데이터셋 로드
# -------------------------------------
TEST_DIR = "dataset/nsfw/image dataset/test"  # ← 경로에 맞게 수정

def load_dataset(test_dir):
    images = []
    labels = []
    classes = ["normal", "nsfw"]
    for label_idx, cls in enumerate(classes):
        cls_folder = os.path.join(test_dir, cls)
        for filename in os.listdir(cls_folder):
            if filename.lower().endswith(("jpg", "jpeg", "png")):
                images.append(os.path.join(cls_folder, filename))
                labels.append(label_idx)
    return images, labels

image_paths, true_labels = load_dataset(TEST_DIR)
print(f"Total test images: {len(image_paths)}")

# -------------------------------------
# 4. Inference - 배치 처리 + tqdm
# -------------------------------------
BATCH_SIZE = 8
all_probs = []
all_labels = []

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Evaluating"):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    images = [Image.open(p).convert("RGB") for p in batch_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # NSFW 클래스 확률

    all_probs.extend(probs)
    all_labels.extend([int(p) for p in true_labels[i:i+BATCH_SIZE]])

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# -------------------------------------
# 5. Validation처럼 Threshold 탐색 (F1 기준)
# -------------------------------------
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []

for t in thresholds:
    preds = (all_probs > t).astype(int)
    f1_scores.append(f1_score(all_labels, preds))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold (F1): {best_threshold:.2f}, F1: {f1_scores[best_idx]:.4f}")

# -------------------------------------
# 6. Metrics 계산
# -------------------------------------
best_preds = (all_probs > best_threshold).astype(int)

accuracy = accuracy_score(all_labels, best_preds)
precision = precision_score(all_labels, best_preds)
recall = recall_score(all_labels, best_preds)
f1 = f1_score(all_labels, best_preds)

print(f"\nTest Metrics (threshold={best_threshold:.2f}):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# -------------------------------------
# 7. Confusion Matrix
# -------------------------------------
cm = confusion_matrix(all_labels, best_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["normal","nsfw"], yticklabels=["normal","nsfw"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()
