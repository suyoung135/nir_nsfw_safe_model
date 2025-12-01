# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
from ultralytics import YOLO
from tqdm import tqdm
from typing import Tuple, Optional, List
from scipy.optimize import linear_sum_assignment # 헝가리안 알고리즘 추가

# ==============================================================================
# ⚙️ 설정 변수 (사용 환경에 맞게 반드시 수정하세요!)
# ==============================================================================
# 1. MobileSAM 설정
SAM_MODEL_TYPE = "vit_t"
SAM_CHECKPOINT_PATH = "MobileSAM/weights/mobile_sam.pt"

# 2. YOLO Pose 모델 설정 (RGB, NIR 모델 경로가 동일하다고 가정)
YOLO_RGB_MODEL_PATH = "yolo_pose_finetune/nsfw_keypoint_run_012/weights/best.pt"
YOLO_NIR_MODEL_PATH = "yolo_pose_finetune/nsfw_keypoint_run_012/weights/best.pt"

# 3. 데이터 경로
RGB_FOLDER = "./dataset/nsfw/image dataset/test/nsfw"
NIR_FOLDER = "./dataset/nsfw/nir img dataset/test/nsfw"
OUTPUT_DIR = "./comparison_result_positive_negative_points_FIXED" # 출력 디렉토리 변경

# 4. 키포인트 설정 (COCO 형식 기준)
# 제외할 키포인트 ID (Negative Point): 얼굴(0-4), 손목(9, 10)
EXCLUDE_KP_IDS = [0, 1, 2, 3, 4, 9, 10] 
# 포함할 키포인트 ID (Positive Point): 어깨(5, 6), 엉덩이(11, 12)
POSITIVE_KP_IDS = [5, 6, 11, 12] 
# 키포인트 신뢰도 임계값
KP_CONF_THRESHOLD = 0.2 # 키포인트가 유효하다고 판단하는 최소 신뢰도
CONF_THRESHOLD = 0.3 # YOLO Box 검출 임계값

# 5. 기타 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 🚨 매칭에 실패한 쌍에 할당할 최대 비용 (calculate_cost_matrix와 통일)
MAX_COST = 1e8 

# ==============================================================================
# 🤖 YOLO 모델 로드 (전역 변수)
# ==============================================================================
YOLO_MODEL_LOADED = True
try:
    yolo_rgb_model = YOLO(YOLO_RGB_MODEL_PATH)
    yolo_nir_model = YOLO(YOLO_NIR_MODEL_PATH)
    print("YOLO Pose 모델 로드 성공.")
except Exception as e:
    print(f"🚨 오류: YOLO 모델 로드 실패. 경로를 확인하세요: {e}")
    YOLO_MODEL_LOADED = False
    yolo_rgb_model, yolo_nir_model = None, None

# ==============================================================================
# 🛠️ 헬퍼 함수
# ==============================================================================

def combine_grayscale_nir(nir_img_gray: np.ndarray) -> np.ndarray:
    """NIR(1채널, 0-255)을 3채널 (0-255, uint8)로 복제"""
    return np.stack([nir_img_gray] * 3, axis=-1)

def load_data(filename: str, rgb_dir: str, nir_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    RGB와 NIR 이미지를 로드하고, NIR을 RGB 크기에 맞춰 3채널로 변환하여 반환합니다.
    """
    rgb_path = os.path.join(rgb_dir, filename)
    nir_path = os.path.join(nir_dir, filename)
    
    # 1. RGB 로드 (기준)
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None: 
        tqdm.write(f"⚠️ RGB load failed (None): {filename}")
        return None, None
        
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # 🌟 A. RGB 크기 유효성 검사 🌟
    h, w = rgb_img.shape[:2]
    if w <= 0 or h <= 0:
        tqdm.write(f"🚨 RGB image has zero dimensions ({w}x{h}): {filename}")
        return None, None
    
    # 2. NIR 로드 
    nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    if nir_img is None: 
        tqdm.write(f"⚠️ NIR load failed (None): {filename}")
        return None, None

    # 🌟 B. NIR 크기 유효성 검사 🌟
    h_nir, w_nir = nir_img.shape[:2]
    if w_nir <= 0 or h_nir <= 0:
        tqdm.write(f"🚨 NIR image has zero dimensions ({w_nir}x{h_nir}): {filename}")
        return None, None
        
    # 3. NIR 이미지를 RGB 크기에 맞춰 조정
    if (h_nir, w_nir) != (h, w):
        nir_img_resized = nir_img
    else:
        target_size = (int(w), int(h))
        
        # 🌟🌟🌟 모든 예외를 포괄하는 try-except로 최종 방어 🌟🌟🌟
        try:
            nir_img_resized = cv2.resize(nir_img, target_size, interpolation=cv2.INTER_CUBIC) 
            
        # 🚨 cv2.error뿐만 아니라 일반 Exception도 포착하여 안전하게 건너뜁니다.
        except Exception as e:
            tqdm.write(f"🚨 FINAL FIX (All Exceptions): {filename} - Resize failed (dsize: {w}x{h}). Error: {type(e).__name__}: {e}")
            return None, None
        
    # 4. 3채널 복제 (YOLO/SAM 입력용)
    nir_img_3ch = combine_grayscale_nir(nir_img_resized)
    
    return rgb_img, nir_img_3ch

def get_all_detections(yolo_model: YOLO, image: np.ndarray, conf_thresh: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """YOLO Pose를 실행하여 이미지 내 모든 사람의 유효한 Box와 Keypoints 리스트를 반환"""
    
    # YOLOv8은 image를 리스트로 받지 않아도 내부적으로 배치 추론 가능
    results = yolo_model.predict(image, conf=conf_thresh, classes=[0], verbose=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return [], []
        
    pr = results[0]

    all_boxes_raw = pr.boxes.xyxy.cpu().numpy()
    all_keypoints = pr.keypoints.data.cpu().numpy()
    
    valid_boxes = []
    valid_keypoints = []

    H, W = image.shape[:2]
    MIN_SIZE = 5
    PADDING = 3

    for i in range(len(all_boxes_raw)):
        x1_raw, y1_raw, x2_raw, y2_raw = all_boxes_raw[i]
        kps = all_keypoints[i]
        
        # 1. 최소 크기 검사
        width = x2_raw - x1_raw
        height = y2_raw - y1_raw
        
        if width < MIN_SIZE or height < MIN_SIZE:
            continue

        # 2. 패딩 적용 및 이미지 경계 클리핑
        x1 = np.clip(x1_raw - PADDING, 0, W)
        y1 = np.clip(y1_raw - PADDING, 0, H)
        x2 = np.clip(x2_raw + PADDING, 0, W)
        y2 = np.clip(y2_raw + PADDING, 0, H)
        
        # 3. 패딩 적용 후 다시 최소 크기 검사
        if (x2 - x1) < MIN_SIZE or (y2 - y1) < MIN_SIZE:
            continue
            
        final_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        valid_boxes.append(final_box)
        valid_keypoints.append(kps)
        
    return valid_boxes, valid_keypoints

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """두 마스크의 IoU 계산"""
    if mask1.shape != mask2.shape:
        tqdm.write(f"[FATAL] Mask shape mismatch: {mask1.shape} vs {mask2.shape}")
        return 0.0

    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    if union == 0: return 0.0
    return intersection / union

def calculate_cost_matrix(kps_rgb_list: List[np.ndarray], kps_nir_list: List[np.ndarray]) -> np.ndarray:
    """
    RGB와 NIR 키포인트 리스트 간의 키포인트 L2 거리를 기반으로 비용 행렬을 계산합니다.
    🚨 신뢰도가 낮은 쌍은 MAX_COST를 할당하여 헝가리안 알고리즘 후 무시되도록 합니다.
    """
    cost_matrix = np.zeros((len(kps_rgb_list), len(kps_nir_list)))
    
    for i, kps_rgb in enumerate(kps_rgb_list):
        for j, kps_nir in enumerate(kps_nir_list):
            # 키포인트 좌표 추출 (x, y, v)
            k1 = kps_rgb[:,:3]
            k2 = kps_nir[:,:3]
            
            # conf > KP_CONF_THRESHOLD 인 키포인트만 매칭에 사용
            mask = (k1[:, 2] > KP_CONF_THRESHOLD) & (k2[:, 2] > KP_CONF_THRESHOLD)
            
            if np.sum(mask) == 0:
                # 🌟 수정: 일치하는 유효 키포인트가 없으면 매우 큰 비용 할당 (MAX_COST와 통일)
                dist = MAX_COST 
            else:
                # 일치하는 키포인트의 L2 거리 합계를 비용으로 사용
                dist = np.sum(np.linalg.norm(k1[mask, :2] - k2[mask, :2], axis=1))
                
            cost_matrix[i, j] = dist
            
    return cost_matrix

# --- 시각화 헬퍼 함수 ---
def show_mask(ax, mask, color):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(ax, box, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # 🌟 수정: facecolor='none'으로 투명하게 설정
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2))


# ==============================================================================
# 🚀 메인 실행
# ==============================================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not YOLO_MODEL_LOADED:
        print("🚨 오류: YOLO 모델이 로드되지 않아 프로그램을 종료합니다.")
        return

    # 1. MobileSAM 로드
    print(f"Loading MobileSAM on {DEVICE}...")
    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam.eval()
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"🚨 오류: MobileSAM 모델 로드 실패. 가중치 파일 경로를 확인하세요: {e}")
        return

    image_files = sorted(os.listdir(RGB_FOLDER))
    tqdm.write(f"\nStarting comparison on {len(image_files)} pairs...")

    total_iou = 0
    valid_count = 0

    for filename in tqdm(image_files, desc="Processing Images"):
        # 1. 데이터 로드 및 전처리
        rgb_img, nir_img_3ch = load_data(filename, RGB_FOLDER, NIR_FOLDER)
        
        if rgb_img is None or nir_img_3ch is None:
            tqdm.write(f"[SKIP] Data load failed for: {filename}")
            continue

        h, w = rgb_img.shape[:2]

        try:
            # A. RGB 파이프라인: 모든 사람 검출
            boxes_rgb, kps_rgb_list = get_all_detections(yolo_rgb_model, rgb_img, CONF_THRESHOLD)
            
            # B. NIR 파이프라인: 모든 사람 검출
            boxes_nir, kps_nir_list = get_all_detections(yolo_nir_model, nir_img_3ch, CONF_THRESHOLD)

            # 2. 매칭할 사람이 없는 경우 스킵
            if not boxes_rgb or not boxes_nir:
                tqdm.write(f"[FAIL] No valid person found in one or both images: {filename}. Skipping.")
                continue

            # 3. 헝가리안 알고리즘을 위한 비용 행렬 계산 (키포인트 거리 기반)
            cost_matrix = calculate_cost_matrix(kps_rgb_list, kps_nir_list)
            
            # 4. 최적 매칭 수행
            rgb_indices, nir_indices = linear_sum_assignment(cost_matrix)
            
            num_matched_pairs = 0
            
            # 🌟 매칭된 쌍에 대해 IoU 계산 및 시각화 🌟
            for r_idx, n_idx in zip(rgb_indices, nir_indices):
                # 🌟 수정: 매칭 비용이 MAX_COST보다 크거나 같으면 유효한 쌍이 아님 -> 무시
                if cost_matrix[r_idx, n_idx] >= MAX_COST:
                    continue
                    
                num_matched_pairs += 1

                # 매칭된 RGB 및 NIR 데이터 추출
                box_rgb = boxes_rgb[r_idx]
                kps_rgb = kps_rgb_list[r_idx]
                box_nir = boxes_nir[n_idx]
                kps_nir = kps_nir_list[n_idx]

                # --- SAM Point 프롬프트 준비 (RGB) ---
                rgb_kps_data = kps_rgb[:, :3]
                
                # 1. Negative Points (레이블 0) 추출: 얼굴/손
                valid_neg_kps_rgb = [
                    rgb_kps_data[i, :2] 
                    for i in EXCLUDE_KP_IDS 
                    if i < len(rgb_kps_data) and rgb_kps_data[i, 2] > KP_CONF_THRESHOLD
                ]
                neg_points_rgb = np.array(valid_neg_kps_rgb, dtype=np.float32)
                
                # 2. Positive Points (레이블 1) 추출: 몸통/어깨/엉덩이
                valid_pos_kps_rgb = [
                    rgb_kps_data[i, :2] 
                    for i in POSITIVE_KP_IDS 
                    if i < len(rgb_kps_data) and rgb_kps_data[i, 2] > KP_CONF_THRESHOLD
                ]
                pos_points_rgb = np.array(valid_pos_kps_rgb, dtype=np.float32)

                # 3. Points 합치기
                point_coords_rgb = np.concatenate([pos_points_rgb, neg_points_rgb], axis=0) if len(pos_points_rgb) > 0 or len(neg_points_rgb) > 0 else None
                
                if point_coords_rgb is not None:
                    # Positive (1) + Negative (0) 레이블 생성
                    pos_labels = np.ones(len(pos_points_rgb), dtype=np.int64)
                    neg_labels = np.zeros(len(neg_points_rgb), dtype=np.int64)
                    point_labels_rgb = np.concatenate([pos_labels, neg_labels], axis=0)
                else:
                    point_labels_rgb = None


                # --- SAM Point 프롬프트 준비 (NIR) ---
                nir_kps_data = kps_nir[:, :3]
                
                valid_neg_kps_nir = [
                    nir_kps_data[i, :2] 
                    for i in EXCLUDE_KP_IDS 
                    if i < len(nir_kps_data) and nir_kps_data[i, 2] > KP_CONF_THRESHOLD
                ]
                neg_points_nir = np.array(valid_neg_kps_nir, dtype=np.float32)
                
                valid_pos_kps_nir = [
                    nir_kps_data[i, :2] 
                    for i in POSITIVE_KP_IDS 
                    if i < len(nir_kps_data) and nir_kps_data[i, 2] > KP_CONF_THRESHOLD
                ]
                pos_points_nir = np.array(valid_pos_kps_nir, dtype=np.float32)

                point_coords_nir = np.concatenate([pos_points_nir, neg_points_nir], axis=0) if len(pos_points_nir) > 0 or len(neg_points_nir) > 0 else None
                
                if point_coords_nir is not None:
                    pos_labels = np.ones(len(pos_points_nir), dtype=np.int64)
                    neg_labels = np.zeros(len(neg_points_nir), dtype=np.int64)
                    point_labels_nir = np.concatenate([pos_labels, neg_labels], axis=0)
                else:
                    point_labels_nir = None
                    
                # --- SAM 세그멘테이션 실행 (RGB) ---
                predictor.set_image(rgb_img)
                box_input_rgb = box_rgb.astype(np.float32)[None, :]
                
                masks_rgb_full, _, _ = predictor.predict(
                    box=box_input_rgb, 
                    point_coords=point_coords_rgb, 
                    point_labels=point_labels_rgb, 
                    multimask_output=False
                )
                mask_rgb_final = masks_rgb_full[0].astype(np.uint8)

                # --- SAM 세그멘테이션 실행 (NIR) ---
                predictor.set_image(nir_img_3ch)
                box_input_nir = box_nir.astype(np.float32)[None, :]
                
                masks_nir_full, _, _ = predictor.predict(
                    box=box_input_nir, 
                    point_coords=point_coords_nir, 
                    point_labels=point_labels_nir, 
                    multimask_output=False
                )
                mask_nir_final = masks_nir_full[0].astype(np.uint8)

                # --- 비교 및 결과 누적 ---
                iou = calculate_iou(mask_rgb_final, mask_nir_final)
                total_iou += iou
                valid_count += 1
                
                # 시각화 저장 조건 (각 쌍의 결과를 별도 저장)
                if iou < 0.8 or num_matched_pairs <= 2: 
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    # Plot 1: RGB
                    axes[0].imshow(rgb_img)
                    show_mask(axes[0], mask_rgb_final, color=np.array([0, 1, 0, 0.6]))
                    show_box(axes[0], box_rgb, color='green')
                    axes[0].set_title(f"RGB Person {r_idx+1} (IoU: {iou:.4f})")

                    # Plot 2: NIR
                    axes[1].imshow(nir_img_3ch)
                    show_mask(axes[1], mask_nir_final, color=np.array([1, 0, 0, 0.6]))
                    show_box(axes[1], box_nir, color='red')
                    axes[1].set_title(f"NIR Person {n_idx+1}")

                    # Plot 3: Overlap
                    axes[2].imshow(mask_rgb_final, cmap='Greens', alpha=0.5)
                    axes[2].imshow(mask_nir_final, cmap='Reds', alpha=0.5)
                    axes[2].set_title(f"Mask Overlap")
                    
                    # Plot 4: Difference
                    diff = np.logical_xor(mask_rgb_final, mask_nir_final)
                    axes[3].imshow(diff, cmap='binary')
                    axes[3].set_title("Pixel Difference (XOR)")

                    for ax in axes: ax.axis('off')
                    plt.suptitle(f"File: {filename} | Matched Pair (RGB:{r_idx+1} <-> NIR:{n_idx+1}) | Final IoU: {iou:.4f}", fontsize=14)
                    plt.savefig(os.path.join(OUTPUT_DIR, f"compare_{filename}_pair{num_matched_pairs}.png"), bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)

        except Exception as e:
            tqdm.write(f"🚨 Critical Error processing {filename}: {e}. Skipping.")
            continue

    # 최종 결과 요약
    if valid_count > 0:
        tqdm.write("\n" + "="*70)
        tqdm.write(f"✅ Comparison Finished on {valid_count} total matched pairs.")
        tqdm.write(f"📊 **(Positive/Negative Point 적용 최종 평균 IoU)**: {total_iou / valid_count:.4f}")
        tqdm.write("="*70)
    else:
        tqdm.write("처리된 유효한 이미지 쌍이 없습니다. 경로, YOLO 모델 성능, 또는 파일 형식을 확인해주세요.")

if __name__ == "__main__":
    main()