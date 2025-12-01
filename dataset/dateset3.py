import os
import shutil
import random

# 1. 파일명을 기준으로 visible과 infrared 폴더에서 이름이 동일한 파일을 찾기
def match_files(visible_dir, infrared_dir):
    visible_files = os.listdir(visible_dir)
    infrared_files = os.listdir(infrared_dir)
    
    # 이름이 동일한 파일을 찾기 위한 세트로 변환
    visible_set = {f[:6] for f in visible_files if f.endswith(('.jpg', '.jpeg', '.png'))}
    infrared_set = {f[:6] for f in infrared_files if f.endswith(('.jpg', '.jpeg', '.png'))}
    
    # visible과 infrared 모두에 존재하는 파일들 찾기
    matched_files = visible_set & infrared_set  # 이름이 동일한 파일 이름만 남겨둠
    
    return matched_files

# 2. 2000장씩 랜덤으로 뽑기
def get_images_to_copy(matched_files, total_images_per_folder):
    # 랜덤으로 2000장씩 선택
    selected_files = random.sample(sorted(matched_files), total_images_per_folder)
    
    return selected_files

# 3. 파일 복사 함수
def copy_files(selected_files, visible_dir, infrared_dir, rgb_dir, nir_dir):
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)  # rgb 폴더가 없다면 생성
    if not os.path.exists(nir_dir):
        os.makedirs(nir_dir)  # nir 폴더가 없다면 생성

    # visible 파일을 rgb 폴더로 복사
    for file_name in selected_files:
        visible_src_path = os.path.join(visible_dir, f"{file_name}.png")
        visible_dest_path = os.path.join(rgb_dir, f"d{file_name}.jpg")
        shutil.copy(visible_src_path, visible_dest_path)
        print(f"복사: {visible_src_path} -> {visible_dest_path}")

    # infrared 파일을 nir 폴더로 복사
    for file_name in selected_files:
        infrared_src_path = os.path.join(infrared_dir, f"{file_name}.png")
        infrared_dest_path = os.path.join(nir_dir, f"d{file_name}.jpg")
        shutil.copy(infrared_src_path, infrared_dest_path)
        print(f"복사: {infrared_src_path} -> {infrared_dest_path}")

# 4. 실행 부분
def main():
    visible_dir = 'dataset/train_A'  # visible 폴더 경로
    infrared_dir = 'dataset/train_B'  # infrared 폴더 경로
    rgb_dir = 'dataset/train/rgb'  # 복사할 rgb 폴더 경로
    nir_dir = 'dataset/train/nir'  # 복사할 nir 폴더 경로
    total_images_per_folder = 2000  # 각 폴더에서 복사할 총 이미지 수

    # 파일 매칭
    matched_files = match_files(visible_dir, infrared_dir)
    
    if not matched_files:
        print("매칭된 파일이 없습니다.")
        return
    
    # 2000장씩 랜덤으로 복사할 이미지 선택
    selected_files = get_images_to_copy(matched_files, total_images_per_folder)
    
    # 파일 복사
    copy_files(selected_files, visible_dir, infrared_dir, rgb_dir, nir_dir)
    print("파일 복사가 완료되었습니다.")

if __name__ == '__main__':
    main()
