import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 깨진 이미지 자동 허용
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from transformers import default_data_collator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# 1. 설정
# ----------------------------
MODEL_NAME = "models/nsfw_classfier"
DATA_DIR = "./dataset/nsfw/nir img dataset/train"  # Train 데이터 경로
TEST_DIR = "./dataset/nsfw/nir img dataset/test"   # Test 데이터 경로
NUM_LABELS = 2
BATCH_SIZE = 8
NUM_EPOCHS = 5
VAL_RATIO = 0.2   # Train에서 Validation으로 자동 split
OUTPUT_DIR = "./nsfw_nir_finetuned"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# 2. Dataset 정의
# ----------------------------
class NSFWDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.processor = processor
        self.images = []
        self.labels = []
        self.classes = ["normal", "nsfw"]
        for label_idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(data_dir, cls)
            if not os.path.exists(cls_folder):
                continue
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(("jpg","jpeg","png")):
                    path = os.path.join(cls_folder, fname)
                    try:
                        img = Image.open(path)
                        img.verify()  # 이미지 무결성 확인
                        self.images.append(path)
                        self.labels.append(label_idx)
                    except:
                        continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        return inputs

# ----------------------------
# 3. Train → Validation 자동 split
# ----------------------------
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
full_dataset = NSFWDataset(DATA_DIR, processor)

val_size = int(len(full_dataset) * VAL_RATIO)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ----------------------------
# 4. 모델 로드
# ----------------------------
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)

# ----------------------------
# 5. Metrics 정의
# ----------------------------
def compute_metrics(p, threshold=0.4):
    logits, labels = p
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    preds = (probs[:,1] > threshold).long()
    labels = torch.tensor(labels)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ----------------------------
# 6. Trainer 설정
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=lambda p: compute_metrics(p, threshold=0.4),
)

# ----------------------------
# 7. Fine-Tuning 실행
# ----------------------------
trainer.train()

# ----------------------------
# 8. Fine-Tuned 모델 저장
# ----------------------------
final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_dir)
processor.save_pretrained(final_model_dir)
print(f"Fine-tuned model saved at {final_model_dir}")

# ----------------------------
# 9. Validation으로 최적 Threshold 탐색 (F1 기준)
# ----------------------------
def get_probs_labels(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_labels, all_probs = [], []
    model.eval()
    for batch in tqdm(loader, desc="Collecting probs"):
        imgs = {k:v.to(device) for k,v in batch.items() if k!="labels"}
        labels = batch["labels"].numpy()
        with torch.no_grad():
            logits = model(**imgs).logits
            probs = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()  # NSFW 확률
        all_labels.extend(labels)
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_probs)

val_labels, val_probs = get_probs_labels(val_dataset)

thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []
for t in thresholds:
    preds = (val_probs > t).astype(int)
    f1_scores.append(f1_score(val_labels, preds))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold on Validation (F1): {best_threshold:.2f}, F1: {f1_scores[best_idx]:.4f}")

# ----------------------------
# 10. ROC & PR Curve 시각화
# ----------------------------
fpr, tpr, _ = roc_curve(val_labels, val_probs)
precision, recall, _ = precision_recall_curve(val_labels, val_probs)
roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR curve (AUC={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# ----------------------------
# 11. Test 평가
# ----------------------------
test_dataset = NSFWDataset(TEST_DIR, processor)

def evaluate_dataset(dataset, threshold):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_labels, all_preds = [], []
    for batch in tqdm(loader, desc="Test Eval"):
        imgs = {k:v.to(device) for k,v in batch.items() if k!="labels"}
        labels = batch["labels"].numpy()
        with torch.no_grad():
            logits = model(**imgs).logits
            probs = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
        preds = (probs > threshold).astype(int)
        all_labels.extend(labels)
        all_preds.extend(preds)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\nTest Metrics (threshold={threshold:.2f}):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["normal","nsfw"], yticklabels=["normal","nsfw"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Test Confusion Matrix")
    plt.show()

evaluate_dataset(test_dataset, best_threshold)
