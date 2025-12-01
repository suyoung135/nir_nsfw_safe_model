import os
import shutil
import random

# 1. 파일명을 기준으로 같은 id와 사진번호를 매칭하기
def match_files(visible_dir, infrared_dir):
    matched_files = {}
    visible_files = os.listdir(visible_dir)
    infrared_files = os.listdir(infrared_dir)
    
    # visible과 infrared 폴더 내 파일 리스트를 순회하며 매칭
    for visible_file in visible_files:
        if visible_file.endswith(('.jpg', '.jpeg', '.png')):
            id_part = visible_file[:2]  # id는 파일명 앞 2자리
            file_base = visible_file[:6]  # id + 사진번호 (예: 01_0001)
            # infrared 폴더에서 같은 파일 이름을 찾음
            if visible_file in infrared_files:
                if id_part not in matched_files:
                    matched_files[id_part] = {'visible': [], 'infrared': []}
                matched_files[id_part]['visible'].append(visible_file)
                matched_files[id_part]['infrared'].append(f"{file_base}.jpg")
    
    return matched_files

# 2. 각 폴더에서 구를  뽑기 위한 비율 계산 후 파일 선택
def get_images_to_copy(matched_files, total_images_per_folder):
    selected_files_visible = []
    selected_files_infrared = []

    # 각 id별로 뽑을 수 있는 파일 수를 계산 (전체에서 800장씩 뽑기 위해 비율 계산)
    total_files = sum(len(files['visible']) for files in matched_files.values())
    
    # 각 id별로 뽑을 수 있는 파일 수 계산 (비율에 맞게)
    id_counts = {id_part: len(files['visible']) for id_part, files in matched_files.items()}
    total_files_to_select = total_images_per_folder  
    
    # 각 id에 대해 몇 개의 이미지를 뽑을지 비율을 맞춰서 결정
    id_selection_count = {}
    remaining_files = total_files_to_select

    for id_part, count in id_counts.items():
        id_selection_count[id_part] = int(count / total_files * total_files_to_select)
        remaining_files -= id_selection_count[id_part]
    
    # 남은 갯수를 마지막 id에 추가
    last_id = list(id_selection_count.keys())[-1]
    id_selection_count[last_id] += remaining_files

    # 선택된 파일들 저장
    for id_part, selected_count in id_selection_count.items():
        visible_files = matched_files[id_part]['visible']
        infrared_files = matched_files[id_part]['infrared']
        
        # 각 id별로 선택된 수만큼 랜덤으로 파일을 선택
        selected_visible_files = random.sample(visible_files, selected_count)
        selected_infrared_files = random.sample(infrared_files, selected_count)
        
        selected_files_visible.extend(selected_visible_files)
        selected_files_infrared.extend(selected_infrared_files)
    
    print(f"{len(selected_files_visible)}와{len(selected_files_infrared)}")
    
    return selected_files_visible, selected_files_infrared

# 3. 파일 복사 함수
def copy_files(selected_files_visible, selected_files_infrared, visible_dir, infrared_dir, rgb_dir, nir_dir):
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)  # rgb 폴더가 없다면 생성
    if not os.path.exists(nir_dir):
        os.makedirs(nir_dir)  # nir 폴더가 없다면 생성

    # visible 파일을 rgb 폴더로 복사
    for visible_file in selected_files_visible:
        visible_src_path = os.path.join(visible_dir, visible_file)
        visible_dest_path = os.path.join(rgb_dir, visible_file)
        shutil.copy(visible_src_path, visible_dest_path)

    # infrared 파일을 nir 폴더로 복사
    for infrared_file in selected_files_infrared:
        infrared_src_path = os.path.join(infrared_dir, infrared_file)
        infrared_dest_path = os.path.join(nir_dir, infrared_file)
        shutil.copy(infrared_src_path, infrared_dest_path)

# 4. 실행 부분
def main():
    visible_dir = 'dataset/train_A'  # visible 폴더 경로
    infrared_dir = 'dataset/train_B'  # infrared 폴더 경로
    rgb_dir = 'dataset/train/rgb'  # 복사할 rgb 폴더 경로
    nir_dir = 'dataset/train/nir'  # 복사할 nir 폴더 경로
    total_images_per_folder = 2000  # 각 폴더에서 복사할 이미지 수

    # 파일 매칭
    matched_files = match_files(visible_dir, infrared_dir)
    
    if not matched_files:
        print("매칭된 파일이 없습니다.")
        return
    
    selected_files_visible, selected_files_infrared = get_images_to_copy(matched_files, total_images_per_folder)
    
    # 파일 복사
    copy_files(selected_files_visible, selected_files_infrared, visible_dir, infrared_dir, rgb_dir, nir_dir)
    print("파일 복사가 완료되었습니다.")

if __name__ == '__main__':
    main()
