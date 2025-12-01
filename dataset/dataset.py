import os
import shutil
import random

orig_root = "dataset/ir"
new_root = "dataset/train"
#sysu-mm01에서  nir 및 rgb 데이터 추출

# 변환 후 저장 폴더
for modality in ['rgb', 'nir']:
    os.makedirs(os.path.join(new_root, modality), exist_ok=True)

def get_id_images(cam_folder):
    """cam_folder 내 ID별 이미지 리스트 반환"""
    cam_path = os.path.join(orig_root, cam_folder)
    id_dict = {}
    for pid in os.listdir(cam_path):
        pid_path = os.path.join(cam_path, pid)
        if os.path.isdir(pid_path):
            imgs = sorted(os.listdir(pid_path))
            id_dict[pid] = [os.path.join(pid_path, img) for img in imgs]
    return id_dict


def get_missing_id_images(rgb_cams, missing_ids):
    """missing_ids에 해당하는 ID에 대해 cam1, cam2에서 이미지를 가져오기"""
    rgb_dicts = [get_id_images(cam) for cam in rgb_cams]  # cam1, cam2에서 이미지 불러오기

    # cam1, cam2의 이미지 목록을 하나로 합치기
    missing_images = []
    for pid in missing_ids:
        # cam1, cam2에서 해당 ID의 이미지 리스트 가져오기
        rgb_imgs_cam1 = rgb_dicts[0].get(pid, [])
        rgb_imgs_cam2 = rgb_dicts[1].get(pid, [])
        
        # 두 카메라에서 이미지를 합치기
        missing_images.extend(rgb_imgs_cam1)
        missing_images.extend(rgb_imgs_cam2)

    return missing_images


extra=0

remain_image = []

def make_pairs(rgb_cams, nir_cams, tag):

    """
    rgb_cams: ['cam1', 'cam2']
    nir_cams: ['cam3']
    tag: 'indoor' or 'outdoor'
    """
    global extra
    global remain_image
    print(f"extra1:{extra}")
    print(f"\n[{tag.upper()} 세트 처리 중...]")

    # 각 cam 이미지 불러오기
    rgb_dicts = [get_id_images(c) for c in rgb_cams]
    nir_dicts = [get_id_images(c) for c in nir_cams]

    paired_list = []

    # 모든 NIR 카메라 기준으로
    nir_ids = set().union(*[nir.keys() for nir in nir_dicts])

    rgb_ids = set().union(*[rgb.keys() for rgb in rgb_dicts])
    unique_cam_ids = rgb_ids-nir_ids
    
    missing_image = get_missing_id_images(rgb_cams,unique_cam_ids)
    
    nir_img_cnt = 0
    rgb_img_cnt = 0 
    for pid in nir_ids:
        # NIR 이미지 모으기
        nir_imgs = sum([nir.get(pid, []) for nir in nir_dicts], [])
        if not nir_imgs:
            continue

        # RGB 이미지 모으기 (ID별로 이미지 선택)
        rgb_imgs_cam1 = rgb_dicts[0].get(pid, [])
        rgb_imgs_cam2 = rgb_dicts[1].get(pid, [])
        
        # NIR 이미지 개수 계산
        nir_len = len(nir_imgs)
        half_nir = nir_len // 2

        # cam1, cam2에서 절반씩 뽑기
        rgb_selected_cam1 = random.sample(rgb_imgs_cam1, min(len(rgb_imgs_cam1), half_nir+nir_len%2))
        rgb_selected_cam2 = random.sample(rgb_imgs_cam2, min(len(rgb_imgs_cam2), nir_len - len(rgb_selected_cam1)))

        # 만약 cam2에서 절반 뽑은 게 부족하다면, cam1에서 부족한 만큼 추가로 뽑기
        if len(rgb_selected_cam2) < half_nir and (len(rgb_selected_cam1)+len(rgb_selected_cam2))<nir_len:

            remaining_needed = half_nir - len(rgb_selected_cam2)
            if len(rgb_imgs_cam1)!=0:
                extra_cam1 = random.sample([img for img in rgb_imgs_cam1 if img not in rgb_selected_cam1], remaining_needed)
                rgb_selected_cam2.extend(extra_cam1)

        remain_image.extend(set(rgb_imgs_cam1)-set(rgb_selected_cam1))
        remain_image.extend(set(rgb_imgs_cam2)-set(rgb_selected_cam2))

        rgb_selected = rgb_selected_cam1 + rgb_selected_cam2
        min_len = len(rgb_selected)

        nir_img_cnt+= nir_len
        rgb_img_cnt+= min_len

        # 페어링
        for i in range(nir_len):
            rgb_src = rgb_selected[i] if i < len(rgb_selected) else None
            nir_src = nir_imgs[i]

            rgb_fname = f"id{pid}_{tag}_{os.path.basename(rgb_src) if rgb_src else 'no_rgb'}"
            nir_fname = f"id{pid}_{tag}_{os.path.basename(nir_src)}"

            rgb_dst = os.path.join(new_root, 'rgb', rgb_fname) if rgb_src else None
            nir_dst = os.path.join(new_root, 'nir', nir_fname)
            if rgb_src:
                shutil.copy2(rgb_src, rgb_dst)
            shutil.copy2(nir_src, nir_dst)

            paired_list.append((rgb_dst, nir_dst, pid))

    print(f"{tag} 세트 페어링 완료: {len(paired_list)}개, remain_image:{len(remain_image)}")
    return paired_list


# -------- 실행 부분 --------
indoor_pairs = make_pairs(['cam1', 'cam2'], ['cam3'], tag='indoor')
outdoor_pairs = make_pairs(['cam4', 'cam5'], ['cam6'], tag='outdoor')



def count_images_in_folder(folder_path, image_extensions=None):
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # 폴더 내 모든 파일 목록 가져오기
    files = os.listdir(folder_path)

    # 이미지 파일만 필터링 (파일이 실제 이미지 파일인지 확인)
    image_files = [
        f for f in files if os.path.isfile(os.path.join(folder_path, f)) and
        any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    return len(image_files)

# 예시 사용
nir_folder = os.path.join(new_root, "nir")
rgb_folder = os.path.join(new_root, "rgb")

nir_count = count_images_in_folder(nir_folder)
rgb_count = count_images_in_folder(rgb_folder)

print(f"NIR 폴더 이미지 개수: {nir_count}")
print(f"RGB 폴더 이미지 개수: {rgb_count}")



rest = nir_count-rgb_count

extra_img = random.sample(remain_image,rest)
for i in range(rest):
    rgb_src = extra_img[i]
    rgb_fname = f"random_{i}"
    rgb_dst = os.path.join(new_root, 'rgb', rgb_fname)
    shutil.copy2(rgb_src, rgb_dst)



print(f"\n총 페어링 수: {len(indoor_pairs) + len(outdoor_pairs)}")
print("SYSU-MM01 RGB/NIR 변환 및 indoor/outdoor 페어링 완료!")
