import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import Tuple
from ultralytics import YOLO

# MobileSAM 라이브러리 임포트 (경로 설정이 올바르다고 가정)
from mobile_sam import sam_model_registry, SamPredictor

# ==============================================================================
# ⚙️ 설정 변수 (반드시 수정하세요!)
# ==============================================================================
# 1. 파일 경로 설정
TEST_IMAGE_PATH = "./dataset/nsfw/nir img dataset/test/nsfw/a_3966.png" # 👈 테스트할 RGB 이미지 경로
OUTPUT_PATH = "./sam_yolo_auto_output.png"
SAM_CHECKPOINT_PATH = "MobileSAM/weights/mobile_sam.pt" # MobileSAM 가중치 경로

# 2. 모델 설정
SAM_MODEL_TYPE = "vit_t"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 자동 경계 박스 (Bounding Box) 변수 (YOLO에 의해 자동 설정됩니다)
# [N, 4] 형식 (N: 탐지된 객체 수)
TEST_BOX = np.array([400, 400, 0, 0])

# ==============================================================================
# 🤖 YOLO 모델 로드 (전역 변수)
# ==============================================================================
try:
    # YOLOv8n (나노 버전)은 빠르고 가볍습니다.
    YOLO_MODEL = YOLO("yolo_pose_finetune/nsfw_keypoint_run_012/weights/best.pt")
    print("YOLOv8n 모델 로드 성공.")
except Exception as e:
    print(f"🚨 오류: YOLOv8 모델 로드 실패. 'pip install ultralytics'를 확인하세요: {e}")
    YOLO_MODEL = None


# ==============================================================================
# 🛠️ 헬퍼 함수
# ==============================================================================

def show_mask(ax, mask, color):
    """마스크를 투명하게 이미지 위에 겹쳐서 표시합니다."""
    h, w = mask.shape[-2:]
    # 마스크를 초록색으로 표시 (RGBA)
    color_map = np.array([0/255, 255/255, 0/255, 0.6]) 
    mask_image = mask.reshape(h, w, 1) * color_map.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(ax, box, color='red'):
    """경계 박스를 이미지 위에 표시합니다. [x_min, y_min, x_max, y_max] 형식"""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # 테두리 색상: 빨간색, 채우기 없음, 선 두께 2
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))

def get_yolo_boxes(img_rgb: np.ndarray) -> np.ndarray:
    """
    YOLOv8을 사용하여 이미지 내의 모든 객체를 탐지하고,
    탐지된 모든 경계 박스 [N, 4]를 numpy 배열로 반환합니다.
    """
    if YOLO_MODEL is None:
        H, W, _ = img_rgb.shape
        return np.array([[0, 0, W, H]]) # 탐지 실패 시 전체 이미지 반환

    # 1. YOLO 예측 실행 (신뢰도 40% 이상만 탐지)
    results = YOLO_MODEL(img_rgb, conf=0.4, verbose=False)
    
    all_boxes = []

    for result in results:
        # 2. 결과에서 경계 박스 좌표 추출
        if result.boxes and len(result.boxes) > 0:
            # .boxes.xyxy: [N, 4] 형태 (x_min, y_min, x_max, y_max)
            # 텐서를 CPU로 이동 후 numpy로 변환
            boxes_tensor = result.boxes.xyxy.cpu() 
            boxes_np = boxes_tensor.numpy()
            all_boxes.append(boxes_np)
            
    if all_boxes:
        # 모든 탐지 결과를 하나의 [N, 4] 배열로 합칩니다.
        return np.concatenate(all_boxes, axis=0)
    else:
        # 탐지된 객체가 없으면 이미지 전체 영역을 반환 (Fallback)
        H, W, _ = img_rgb.shape
        print("🚨 경고: YOLO가 객체를 탐지하지 못했습니다. 이미지 전체 박스를 사용합니다.")
        return np.array([[0, 0, W, H]]) # [1, 4] 형태 유지


# ==============================================================================
# 🚀 메인 실행
# ==============================================================================

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"🚨 오류: 테스트 이미지 경로를 찾을 수 없습니다: {TEST_IMAGE_PATH}")
        return

    # 1. 이미지 로드
    img_bgr = cv2.imread(TEST_IMAGE_PATH)
    if img_bgr is None:
        print(f"🚨 오류: 이미지 파일을 로드할 수 없습니다. 파일 형식/손상 여부를 확인하세요.")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # ----------------------------------------------------------------------
    # ⚙️ YOLO를 사용한 자동 경계 박스 설정
    # ----------------------------------------------------------------------
    global TEST_BOX
    TEST_BOX = get_yolo_boxes(img_rgb)
    # ----------------------------------------------------------------------
    
    # 2. MobileSAM 모델 로드
    print(f"Loading MobileSAM on {DEVICE}...")
    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam.eval()
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"🚨 오류: MobileSAM 모델 로드 실패. 가중치 파일 경로를 확인하세요: {e}")
        return

    print(f"Starting prediction using **YOLO-DETECTED** BOXES (N={TEST_BOX.shape[0]}):")
    
    # 3. SAM 예측 실행
    try:
        predictor.set_image(img_rgb)
    
        # ---------------------------------------------------------------------------------
        # 수정: Box 입력 처리
        # 1. TEST_BOX (NumPy 배열)를 PyTorch 텐서로 변환 (CUDA 장치 사용 시 필요)
        # 2. PyTorch 텐서의 데이터 타입을 float32로 명시적으로 지정
        # 3. MobileSAM은 박스 텐서를 내부적으로 처리하므로, 텐서를 직접 전달합니다.
        #    단, MobileSAM/SAM의 predict 함수가 dtype을 강제할 수 있으므로 float32를 사용합니다.
        # ---------------------------------------------------------------------------------
        
        # 텐서로 변환하고 float32 타입 강제
        # 1. NumPy 배열을 float32로 변환 (NumPy 배열에 astype 사용)
        input_box = TEST_BOX.astype(np.float32) 
        
        # 2. DEVICE로 이동시키지 않고, NumPy 배열 그대로 전달
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box, # <--- NumPy 배열 전달
            multimask_output=False
        )
        
    except Exception as e:
        print(f"🚨 오류: SAM 예측 중 오류 발생: {e}")
        return
        
    # 4. 결과 시각화 및 저장
    if masks.shape[0] == 0:
        print("✅ 예측 성공: 마스크가 생성되지 않았습니다 (박스 내에 객체가 없거나 경계가 모호할 수 있습니다).")
        return
        
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    
    # N개의 탐지 결과 반복
    for i in range(masks.shape[0]):
        mask = masks[i]
        score = scores[i]
        box = TEST_BOX[i]
        
        # 마스크와 박스 표시
        show_mask(plt.gca(), mask, color=np.array([0, 255/255, 0, 0.6]))
        show_box(plt.gca(), box, color='red') 
        
        # 박스 상단에 객체 점수 표시 (선택 사항)
        plt.text(box[0], box[1] - 10, f'Score: {score:.2f}', color='red', fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        
        print(f"Box {i+1} - Bounding Box: {box.tolist()}, Score: {score:.4f}")

    plt.title(f"YOLO + MobileSAM Result (Total Objects: {masks.shape[0]})", fontsize=14)
    plt.axis('off')
    plt.savefig(OUTPUT_PATH, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"\n✅ 테스트 완료: 마스크 결과가 '{OUTPUT_PATH}'에 저장되었습니다.")

if __name__ == "__main__":
    main()