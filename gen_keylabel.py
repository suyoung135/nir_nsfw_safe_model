import os
import glob
from ultralytics import YOLO
from tqdm import tqdm
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # 파일 끝이 잘려도 로드하도록 설정
# -----------------------------
# 설정
# -----------------------------
# 1. 이미지가 있는 폴더 경로
IMAGE_DIR = "./dataset/nsfw/image dataset/train/nsfw" 

# 2. 라벨을 저장할 폴더 경로 (자동 생성됨)
LABEL_DIR = "./dataset/nsfw/image dataset/nsfw_labels"

# 3. 라벨링에 사용할 모델
# 주의: 라벨링은 속도보다 '정확도'가 중요하므로, 가능하면 가장 큰 모델(yolov8x-pose.pt)을 사용하는 것이 좋습니다.
# 다운로드가 안 되어 있다면 자동으로 다운로드 됩니다.
MODEL_NAME = "yolov8x-pose.pt" 
CONF_THRESHOLD = 0.3  # 이 값보다 신뢰도가 낮으면 라벨링 하지 않거나 안 보임 처리

# -----------------------------
# 코드 시작
# -----------------------------
def generate_keypoint_labels():
    # GPU 사용 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 모델 로드
    print(f"Loading model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME).to(device)

    # 이미지 파일 리스트 가져오기
    # 지원하는 확장자 추가 가능
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    print(f"Found {len(image_files)} images.")

    # 라벨 저장 디렉토리 생성
    os.makedirs(LABEL_DIR, exist_ok=True)

    # 배치 처리는 복잡해질 수 있어 스트림 방식으로 처리 (메모리 효율적)
    # 11,000장이면 tqdm으로 진행 상황을 보는 것이 필수입니다.
    
    for img_path in tqdm(image_files, desc="Generating Labels"):
        try:
            # 추론 (stream=False로 하여 결과를 바로 받음)
            results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False, device=device)
            result = results[0]

            # 저장할 텍스트 파일 경로 생성
            base_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            txt_path = os.path.join(LABEL_DIR, file_name_without_ext + ".txt")

            label_lines = []

            # 검출된 객체(사람)가 있는 경우에만 처리
            if result.boxes is not None and result.keypoints is not None:
                # 박스 좌표 (Normalized xywh)
                boxes_xywhn = result.boxes.xywhn.cpu().numpy()
                
                # 키포인트 좌표 (Normalized xy)
                kpts_xyn = result.keypoints.xyn.cpu().numpy()
                
                # 키포인트 confidence (이게 없으면 visibility 설정 어려움)
                # conf가 없는 경우도 있을 수 있으니 체크
                if result.keypoints.conf is not None:
                    kpts_conf = result.keypoints.conf.cpu().numpy()
                else:
                    # conf가 없으면 모두 1.0(보임)으로 가정하거나 처리 필요
                    kpts_conf = [[1.0]*17] * len(boxes_xywhn)

                # 사람 한 명씩 반복
                for i in range(len(boxes_xywhn)):
                    # 1. Bounding Box (class_id는 0으로 고정)
                    # format: class x_center y_center width height
                    box = boxes_xywhn[i]
                    line_parts = [0, box[0], box[1], box[2], box[3]]

                    # 2. Keypoints
                    # format: px1 py1 v1 px2 py2 v2 ...
                    # v(visibility): 0=안보임/미검출, 1=가려짐, 2=보임
                    # YOLOv8 학습 시에는 보통 2(visible) 혹은 0(missing)을 주로 사용
                    
                    kpt_xy = kpts_xyn[i]      # (17, 2)
                    kpt_c = kpts_conf[i]      # (17,)

                    for j in range(17):
                        x, y = kpt_xy[j]
                        conf = kpt_c[j]
                        
                        # 좌표가 (0,0)이거나 신뢰도가 너무 낮으면 visibility=0
                        if conf < 0.1 or (x == 0 and y == 0):
                            v = 0
                            x, y = 0.0, 0.0 # 위치도 0으로 초기화
                        else:
                            v = 2 # 2: Visible
                        
                        line_parts.extend([x, y, v])

                    # 라인 생성 (소수점 6자리까지)
                    line_str = " ".join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in line_parts])
                    label_lines.append(line_str)

            # 결과 파일 저장
            # 사람이 없어도 빈 파일은 생성하는 것이 관례일 수 있으나, 
            # 보통 학습 시에는 라벨 파일이 없으면 배경(background) 이미지로 간주함.
            # 여기서는 라벨이 있을 때만 저장합니다.
            if label_lines:
                with open(txt_path, "w") as f:
                    f.write("\n".join(label_lines))

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

    print("\n✅ 라벨 생성 완료!")
    print(f"라벨 파일 저장 위치: {LABEL_DIR}")

if __name__ == "__main__":
    generate_keypoint_labels()