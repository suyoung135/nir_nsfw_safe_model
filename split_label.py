import os
import glob
import random
import shutil
from tqdm import tqdm

# =======================================================
# ⚙️ 설정
# =======================================================
# 1. 원본 데이터 경로 (이전에 사용하셨던 경로 기반)
SOURCE_IMAGE_DIR = "./dataset/nsfw/nir img dataset/train/nsfw"
SOURCE_LABEL_DIR = "./dataset/nsfw/image dataset/nsfw_labels"

# 2. 분할된 데이터를 저장할 최상위 디렉토리
# 이 폴더 안에 train/images, train/labels, val/images, val/labels가 생성됩니다.
DEST_ROOT_DIR = "./yolo_pose_dataset_split"

# 3. 분할 비율 (예: 80% 학습, 20% 검증)
TRAIN_RATIO = 0.8

# 4. Keypoint 정보 (COCO 17 Keypoints)
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# =======================================================

def split_data_and_create_yaml():
    """이미지와 라벨을 train/val로 분할하고 data.yaml을 생성합니다."""
    
    # 1. 이미지 파일 목록 가져오기 (확장자는 .jpg, .png 등으로 가정)
    image_files = glob.glob(os.path.join(SOURCE_IMAGE_DIR, "*.*"))
    if not image_files:
        print("❌ 오류: 원본 이미지 폴더에 파일이 없습니다. 경로를 확인해주세요.")
        return

    # 2. 이미지-라벨 쌍 목록 생성 (라벨 파일이 없는 경우를 대비하여 이미지를 기준으로 처리)
    file_pairs = []
    for img_path in image_files:
        base_name = os.path.basename(img_path)
        name_only = os.path.splitext(base_name)[0]
        label_path = os.path.join(SOURCE_LABEL_DIR, name_only + ".txt")
        
        # 라벨 파일이 존재할 때만 쌍으로 묶음 (라벨이 없는 이미지는 스킵)
        if os.path.exists(label_path):
            file_pairs.append((img_path, label_path))
            
    if not file_pairs:
        print("❌ 오류: 유효한 이미지/라벨 쌍이 없습니다. 라벨 폴더에 파일이 있는지 확인해주세요.")
        return

    print(f"✅ 총 {len(file_pairs)}개의 유효한 이미지-라벨 쌍을 찾았습니다.")

    # 3. 데이터셋 랜덤 셔플 및 분할
    random.seed(42) # 재현성을 위해 시드 고정
    random.shuffle(file_pairs)
    
    split_index = int(len(file_pairs) * TRAIN_RATIO)
    train_set = file_pairs[:split_index]
    val_set = file_pairs[split_index:]
    
    print(f"   -> 학습(Train) 세트: {len(train_set)}개")
    print(f"   -> 검증(Val) 세트: {len(val_set)}개")
    
    # 4. 저장 폴더 구조 생성
    for subset in ['train', 'val']:
        for file_type in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_ROOT_DIR, subset, file_type), exist_ok=True)

    # 5. 파일 이동/복사
    def move_files(dataset, subset_name):
        for img_path, label_path in tqdm(dataset, desc=f"Moving {subset_name} files"):
            # 이미지 파일 이동
            img_dest = os.path.join(DEST_ROOT_DIR, subset_name, 'images', os.path.basename(img_path))
            shutil.copy2(img_path, img_dest) # 원본 유지를 위해 복사(copy2) 사용
            
            # 라벨 파일 이동
            label_dest = os.path.join(DEST_ROOT_DIR, subset_name, 'labels', os.path.basename(label_path))
            shutil.copy2(label_path, label_dest)
            
    move_files(train_set, 'train')
    move_files(val_set, 'val')

    # 6. data.yaml 파일 생성
    yaml_content = f"""
# YOLOv8 Pose Fine-tuning Configuration
path: {os.path.abspath(DEST_ROOT_DIR)}

# Train and Validation directories relative to 'path'
train: train/images
val: val/images

# Classes
nc: 1
names: ['person']

# Keypoints Configuration (COCO 17 Keypoints)
kpt_shape: [{len(KPT_NAMES)}, 3]
kpt_names: {KPT_NAMES}
"""
    yaml_path = os.path.join(DEST_ROOT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print("\n\n🎉 데이터 분할 및 준비 완료!")
    print(f"저장 위치: {os.path.abspath(DEST_ROOT_DIR)}")
    print(f"설정 파일: {yaml_path}")

    return yaml_path

if __name__ == "__main__":
    yaml_path = split_data_and_create_yaml()

    if yaml_path:
        print("\n=============================================")
        print("👇 이제 다음 코드를 실행하여 파인튜닝을 시작하세요.")
        print("=============================================")
        print(f"from ultralytics import YOLO\n")
        print(f"# yolov8x-pose.pt를 다운받아 시작 (가장 큰 모델 권장)\n")
        print(f"model = YOLO('yolov8x-pose.pt')\n")
        print(f"results = model.train(\n")
        print(f"    data='{yaml_path}', \n")
        print(f"    epochs=100, \n")
        print(f"    imgsz=640, \n")
        print(f"    device=0 # GPU 사용 번호 (CPU 사용 시 'cpu')\n")
        print(f")")