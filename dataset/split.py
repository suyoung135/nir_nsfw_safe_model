import os
import random
import shutil
from collections import defaultdict

# -----------------------------
# 설정
# -----------------------------
rgb_dir = 'dataset/train/rgb'
nir_dir = 'dataset/train/nir'
val_rgb_dir = 'dataset/val/rgb'
val_nir_dir = 'dataset/val/nir'
val_ratio = 0.1
seed = 42

os.makedirs(val_rgb_dir, exist_ok=True)
os.makedirs(val_nir_dir, exist_ok=True)

random.seed(seed)

# -----------------------------
# 1. 파일을 ID별로 그룹화
# -----------------------------
def group_by_id(file_list):
    id_dict = defaultdict(list)
    for f in file_list:
        pid = f.split('_')[0]  # id번호 기준
        id_dict[pid].append(f)
    return id_dict

rgb_files = os.listdir(rgb_dir)
nir_files = os.listdir(nir_dir)

rgb_by_id = group_by_id(rgb_files)
nir_by_id = group_by_id(nir_files)

# -----------------------------
# 2. ID별로 val 추출 (nir 기준으로)
# -----------------------------
for pid in nir_by_id.keys():
    nir_imgs = sorted(nir_by_id[pid])
    rgb_imgs = sorted(rgb_by_id.get(pid, []))
    
    # nir 이미지가 없으면 건너뛰기
    if not nir_imgs:
        continue
    
    # 최소 1개의 이미지가 항상 val로 가도록 설정
    n_val = max(1, int(len(nir_imgs) * val_ratio))  
    
    # nir 기준으로 val_indices 추출
    val_indices = random.sample(range(len(nir_imgs)), n_val)

    for idx in val_indices:
        nir_fname = nir_imgs[idx]
        
        # 해당 인덱스의 rgb 이미지가 있으면 val로 이동
        rgb_fname = rgb_imgs[idx] if idx < len(rgb_imgs) else None
        
        # 만약 대응되는 rgb 이미지가 없으면 nir 이미지도 이동하지 않음
        if rgb_fname:
            shutil.move(os.path.join(rgb_dir, rgb_fname), os.path.join(val_rgb_dir, rgb_fname))
            shutil.move(os.path.join(nir_dir, nir_fname), os.path.join(val_nir_dir, nir_fname))

print("RGB/NIR 짝 맞춘 10% val 분리 완료!")
