import os
import glob
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# -----------------------------
# 설정 (두 모델 경로 추가)
# -----------------------------
RGB_FOLDER = "./dataset/nsfw/image dataset/test/nsfw"
NIR_FOLDER = "./dataset/nsfw/nir img dataset/test/nsfw"

# **수정: RGB 전용 모델과 NIR 전용 모델의 경로를 별도로 설정**
# 이 경로는 사용자의 실제 모델 파일 경로에 맞게 변경해야 합니다.
RGB_POSE_MODEL = "yolo_pose_finetune/nsfw_keypoint_run_01/weights/best.pt"  
NIR_POSE_MODEL = "yolo_pose_finetune/nsfw_keypoint_run_01/weights/best.pt"  # NIR 데이터로 학습된 모델

IMG_MAX_SIDE = 1280
CONF_THRESH = 0.3
PCK_THRESH = 0.3
BATCH_SIZE = 8

COCO_KPTS = 17
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# -----------------------------
# GPU 모델 로드 (두 모델을 각각 로드)
# -----------------------------
try:
    rgb_pose_model = YOLO(RGB_POSE_MODEL).to("cuda")
    nir_pose_model = YOLO(NIR_POSE_MODEL).to("cuda")
except Exception as e:
    print(f"Error loading models to CUDA: {e}. Falling back to CPU for both.")
    rgb_pose_model = YOLO(RGB_POSE_MODEL)
    nir_pose_model = YOLO(NIR_POSE_MODEL)
    BATCH_SIZE = 1 # CPU 사용 시 배치 크기 조정

# -----------------------------
# Keypoint 변환 유틸 (변경 없음)
# -----------------------------
def to_coco_keypoints_from_yolov8pose(yolo_keypoints):
    """YOLOv8 pose keypoints(x, y, conf)를 COCO 형식(x, y, v)으로 변환"""
    k = np.array(yolo_keypoints)
    if k.ndim==1: k=k.reshape(-1,3)
    out=[]
    for (x,y,conf) in k:
        # v=2 (보임, conf > 0.01), v=0 (미검출)로 단순화
        v = 2 if conf > 0.01 else 0
        out.extend([float(x),float(y),int(v)])
    
    if len(out) < COCO_KPTS*3:
        out += [0.0,0.0,0]*(COCO_KPTS - len(out)//3)
    return out[:COCO_KPTS*3]

# -----------------------------
# 이미지 읽기 (변경 없음)
# -----------------------------
def load_image_safe(path):
    try:
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        h,w = img.shape[:2]
        scale = 1.0
        if max(h,w) > IMG_MAX_SIDE:
            scale = IMG_MAX_SIDE/float(max(h,w))
            img_resized = cv2.resize(img,(int(w*scale),int(h*scale)))
        else:
            img_resized = img.copy()
            
        return img, img_resized, scale
    except Exception as e:
        # print(f"Warning: {path} 읽기 실패 ({e}), 건너뜀")
        return None, None, 1.0

# -----------------------------
# Keypoint metric 계산 (변경 없음)
# -----------------------------
def compute_metrics_per_kpt(kpts1, kpts2):
    """
    Keypoint 쌍에 대한 Normalized L2, PCK, Consistency 계산.
    L2: v=2 vs v=2 일 때만 계산 (순수 위치 오차 측정).
    PCK/Consistency: v>0 vs v>0 일 때 1점으로 처리 (검출 안정성/품질 측정).
    """
    k1 = np.array(kpts1).reshape(-1,3)
    k2 = np.array(kpts2).reshape(-1,3)
    
    # 정규화 기준 max_dist 계산 (v>0인 모든 Keypoint를 포함)
    valid_kpts = np.concatenate([k1[k1[:,2]>0,:2], k2[k2[:,2]>0,:2]], axis=0)
    if len(valid_kpts) < 2:
        max_dist = 0.0 
    else:
        x_vals = valid_kpts[:,0]
        y_vals = valid_kpts[:,1]
        max_dist = np.sqrt((np.max(x_vals)-np.min(x_vals))**2 + (np.max(y_vals)-np.min(y_vals))**2)

    norm_l2_list=[]
    pck_list=[]
    consistent_list=[] 
    
    for i in range(COCO_KPTS):
        k1_conf = k1[i, 2]
        k2_conf = k2[i, 2]
        
        k1_detected = k1_conf > 0
        k2_detected = k2_conf > 0
        
        k1_v2 = k1_conf == 2
        k2_v2 = k2_conf == 2
        
        # 1. 정규화 기준이 0 (계산 불가)인 경우
        if max_dist == 0.0:
            norm_l2_list.append(np.nan)
            pck_list.append(np.nan)
            consistent_list.append(np.nan)
        
        # 2. L2 계산 (v=2 vs v=2 조건)
        elif k1_v2 and k2_v2:
            dist = np.linalg.norm(k1[i,:2]-k2[i,:2])
            norm_l2 = dist / max_dist
            pck = 1 if dist < (PCK_THRESH*max_dist) else 0 
            
            norm_l2_list.append(norm_l2)
            pck_list.append(pck)
            consistent_list.append(1)
            
        # 3. L2 계산 대상이 아닐 때
        else:
            norm_l2_list.append(np.nan)
            
            if k1_detected and k2_detected:
                dist = np.linalg.norm(k1[i,:2]-k2[i,:2])
                pck = 1 if dist < (PCK_THRESH*max_dist) else 0
                consistent_list.append(1)
            else:
                pck_list.append(0)
                consistent_list.append(0)
            
    return norm_l2_list, pck_list, consistent_list

# -----------------------------
# Folder 평가 (배치 처리 적용)
# -----------------------------
def evaluate_folder_multi():
    rgb_files = sorted(glob.glob(os.path.join(RGB_FOLDER,"*.*")))
    nir_files = sorted(glob.glob(os.path.join(NIR_FOLDER,"*.*")))
    
    if len(rgb_files) != len(nir_files):
        print("Error: RGB/NIR 이미지 수 불일치. 평가를 중단합니다.")
        return
        
    all_norm_l2 = {k:[] for k in KPT_NAMES}
    all_pck = {k:[] for k in KPT_NAMES}
    all_consistent = {k:[] for k in KPT_NAMES} 

    file_pairs = list(zip(rgb_files, nir_files))
    
    for i in tqdm(range(0, len(file_pairs), BATCH_SIZE), desc="Evaluating batches"):
        batch_pairs = file_pairs[i:i + BATCH_SIZE]
        
        # 1. 이미지 로드 및 리사이즈 
        rgb_data = [load_image_safe(p[0]) for p in batch_pairs]
        nir_data = [load_image_safe(p[1]) for p in batch_pairs]
        
        rgb_resized_imgs = [r_img for img, r_img, scale in rgb_data if r_img is not None]
        nir_resized_imgs = [r_img for img, r_img, scale in nir_data if r_img is not None]

        # 2. Keypoint 추출 (RGB는 RGB 전용 모델, NIR은 NIR 전용 모델 사용)
        rgb_pose_results = []
        if rgb_resized_imgs:
            # **수정: RGB 이미지에 RGB 전용 모델 사용**
            rgb_pose_results = rgb_pose_model.predict(source=rgb_resized_imgs, conf=CONF_THRESH, device="cuda" if "cuda" in str(rgb_pose_model.device) else "cpu", verbose=False)
        
        nir_pose_results = []
        if nir_resized_imgs:
            # **수정: NIR 이미지에 NIR 전용 모델 사용**
            nir_pose_results = nir_pose_model.predict(source=nir_resized_imgs, conf=CONF_THRESH, device="cuda" if "cuda" in str(nir_pose_model.device) else "cpu", verbose=False)

        # 3. 배치 결과 처리 (변경 없음)
        rgb_persons_batch = []
        rgb_idx = 0
        for img, img_resized, scale in rgb_data:
            if img_resized is None:
                rgb_persons_batch.append([])
                continue
            
            persons = []
            try:
                pr = rgb_pose_results[rgb_idx]
                boxes = pr.boxes.xyxy.cpu().numpy()
                kpts_all = pr.keypoints.xy.cpu().numpy()
                kconfs_all = pr.keypoints.conf.cpu().numpy()
                
                for j in range(len(boxes)):
                    bbox = boxes[j].tolist()
                    kp_stack = np.concatenate([kpts_all[j], kconfs_all[j].reshape(-1,1)], axis=1)
                    coco = to_coco_keypoints_from_yolov8pose(kp_stack)
                    
                    if scale != 1.0:
                        kps = np.array(coco).reshape(-1,3)
                        kps[:,0:2] = kps[:,0:2] / scale
                        coco = kps.flatten().tolist()
                        bbox=[coord/scale for coord in bbox]
                    persons.append({"bbox":bbox,"coco_keypoints":coco})
            except Exception as e:
                pass
            
            rgb_persons_batch.append(persons)
            rgb_idx += 1


        nir_persons_batch = []
        nir_idx = 0
        for img, img_resized, scale in nir_data:
            if img_resized is None:
                nir_persons_batch.append([])
                continue
                
            persons = []
            try:
                pr = nir_pose_results[nir_idx]
                boxes = pr.boxes.xyxy.cpu().numpy()
                kpts_all = pr.keypoints.xy.cpu().numpy()
                kconfs_all = pr.keypoints.conf.cpu().numpy()
                
                for j in range(len(boxes)):
                    bbox = boxes[j].tolist()
                    kp_stack = np.concatenate([kpts_all[j], kconfs_all[j].reshape(-1,1)], axis=1)
                    coco = to_coco_keypoints_from_yolov8pose(kp_stack)
                    
                    if scale != 1.0:
                        kps = np.array(coco).reshape(-1,3)
                        kps[:,0:2] = kps[:,0:2] / scale
                        coco = kps.flatten().tolist()
                        bbox=[coord/scale for coord in bbox]
                    persons.append({"bbox":bbox,"coco_keypoints":coco})
            except Exception as e:
                pass
            
            nir_persons_batch.append(persons)
            nir_idx += 1
            
        
        # 4. 이미지 쌍별 매칭 및 지표 계산 (변경 없음: 두 모델의 키포인트 일관성 측정)
        for rgb_persons, nir_persons in zip(rgb_persons_batch, nir_persons_batch):
            if not rgb_persons or not nir_persons:
                continue

            # Hungarian matching
            cost_matrix = []
            for rp in rgb_persons:
                row=[]
                for np_ in nir_persons:
                    k1 = np.array(rp["coco_keypoints"]).reshape(-1,3)
                    k2 = np.array(np_["coco_keypoints"]).reshape(-1,3)
                    mask = (k1[:,2]>0) & (k2[:,2]>0)
                    if np.sum(mask) == 0:
                        dist = 1e6  
                    else:
                        # 일치하는 키포인트의 L2 거리를 비용으로 사용
                        dist = np.sum(np.linalg.norm(k1[mask,:2]-k2[mask,:2],axis=1))
                    row.append(dist)
                cost_matrix.append(row)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 5. 매칭된 쌍에 대해 지표 계산 
            for r_idx, n_idx in zip(row_ind, col_ind):
                if cost_matrix[r_idx][n_idx] >= 1e6: continue  
                
                k1 = rgb_persons[r_idx]["coco_keypoints"]
                k2 = nir_persons[n_idx]["coco_keypoints"]
                
                # RGB 모델 출력 (k1)과 NIR 모델 출력 (k2) 간의 일관성 지표 계산
                norm_l2_list, pck_list, consistent_list = compute_metrics_per_kpt(k1,k2)
                
                for k,name in enumerate(KPT_NAMES):
                    if not np.isnan(norm_l2_list[k]):
                        all_norm_l2[name].append(norm_l2_list[k])
                    
                    if not np.isnan(consistent_list[k]): 
                        all_pck[name].append(pck_list[k])
                        all_consistent[name].append(consistent_list[k])


    # 최종 결과 출력 부분은 변경 없음. (두 모델 출력 간의 일관성 평균을 출력)
    print("\n===== 두 도메인 최적 모델 간의 Keypoint 일관성 평균 =====")
    for name in KPT_NAMES:
        valid_l2_count = len(all_norm_l2[name])
        total_eval_count = len(all_consistent[name])

        if total_eval_count > 0:
            mean_pck = np.mean(all_pck[name])
            mean_consistent = np.mean(all_consistent[name])

            if valid_l2_count > 0:
                mean_l2 = np.mean(all_norm_l2[name])
                print(f"**{name}**: Mean L2={mean_l2:.3f} (Valid Pairs: {valid_l2_count}), Mean PCK={mean_pck:.3f}, Consistency={mean_consistent:.3f} (Total: {total_eval_count})")
            else:
                print(f"**{name}**: Mean L2=N/A (Valid Pairs: 0), Mean PCK={mean_pck:.3f}, Consistency={mean_consistent:.3f} (Total: {total_eval_count})")
        else:
            print(f"**{name}**: 평가 불가 (No valid person pairs found)")

if __name__=="__main__":
    evaluate_folder_multi()