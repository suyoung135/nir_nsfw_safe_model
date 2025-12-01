import torch
from ultralytics import YOLO
import os

# =======================================================
# ⚙️ 설정 (이전 스크립트의 결과물 경로를 사용)
# =======================================================
# 1. 학습에 사용할 데이터셋 YAML 파일 경로 (이전 스크립트에서 생성됨)
# 파일 경로가 다를 경우 반드시 수정해야 합니다.
DATA_YAML_PATH = './yolo_pose_dataset_split/data.yaml'

# 2. 파인튜닝에 사용할 사전 학습된 모델 (가장 큰 모델 권장)
MODEL_NAME = 'yolov8m-pose.pt' 

# 3. 학습 하이퍼파라미터
EPOCHS = 100     # 에폭 수 (11,000장 기준, 100~300 사이에서 시작 권장)
IMG_SIZE = 640   # 이미지 입력 크기 (GPU 메모리 상황에 따라 1280 등으로 변경 가능)
BATCH_SIZE = 16   # 배치 사이즈 (GPU 메모리에 맞게 4, 8, 16 등으로 조절)

# 4. GPU 설정
# torch.cuda.current_device()를 사용하여 현재 GPU 번호를 가져오거나, 
# GPU 0번을 사용하도록 설정. CPU만 사용하려면 'cpu'로 변경.
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# 5. 결과 저장 폴더 이름 설정
PROJECT_NAME = 'yolo_pose_finetune'
RUN_NAME = 'nsfw_keypoint_run_01'
# =======================================================


def run_finetuning():
    """
    YOLOv8 Pose 모델을 커스텀 데이터셋으로 파인튜닝합니다.
    """
    
    # 1. GPU 사용 가능 여부 확인 및 정보 출력
    if DEVICE != 'cpu':
        print(f"Using device: CUDA device {DEVICE}")
        if BATCH_SIZE * len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')) > 32:
             print("⚠️ 주의: 배치 사이즈가 너무 크면 메모리 부족이 발생할 수 있습니다.")
    else:
        print("Using device: CPU (학습 속도가 매우 느릴 수 있습니다)")
    
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ 오류: {DATA_YAML_PATH} 파일을 찾을 수 없습니다. 경로를 확인하거나 데이터 분할 스크립트를 먼저 실행하세요.")
        return

    # 2. 모델 로드 (자동으로 다운로드됨)
    print(f"\nLoading model: {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 3. 학습 시작
    print("\n🚀 파인튜닝을 시작합니다...")
    results = model.train(
        data=DATA_YAML_PATH,     # 데이터셋 설정 파일
        epochs=EPOCHS,           # 학습할 에폭 수
        imgsz=IMG_SIZE,          # 이미지 크기
        batch=BATCH_SIZE,        # 배치 사이즈
        device=DEVICE,           # 사용할 장치 (GPU/CPU)
        project=PROJECT_NAME,    # 결과 저장할 폴더명
        name=RUN_NAME,           # 결과 폴더 내의 실행 폴더명
        patience=50,             # 50 에폭 동안 성능 개선 없으면 조기 종료            
        dropout=0.1              # 드롭아웃 설정 (과적합 방지, 선택 사항)
    )

    print("\n✅ 파인튜닝 완료!")
    print(f"결과는 {os.path.join('runs', 'pose', PROJECT_NAME, RUN_NAME)} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    run_finetuning()