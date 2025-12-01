import torch
import os
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms

# 파일 끝이 잘려도 로드하도록 설정 (손상된 파일 처리)
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# ----------------------------------------------------
# 1. infer_from_torchscript 함수 (원본 크기 복원 로직 추가)
# ----------------------------------------------------
def infer_from_torchscript(script_path: str, input_dir: str, output_dir: str, image_size: int = 256):
    """
    TorchScript (.pt) 모델을 로드하여 특정 폴더의 RGB 이미지를 NIR 이미지로 변환 후 저장합니다.
    **생성된 이미지를 원본 RGB 이미지 크기로 복원하여 저장합니다.**
    
    Args:
        script_path (str): 로드할 TorchScript 파일 경로 (예: 'deploy_netG.pt')
        input_dir (str): 변환할 원본 RGB 이미지가 있는 폴더 경로
        output_dir (str): 변환된 NIR 이미지를 저장할 폴더 경로
        image_size (int): 모델 학습 시 사용된 입력/출력 이미지 크기 (기본값 256)
    """
    # 1. 디바이스 설정 및 출력 폴더 생성
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    print(f"🛠️ 실행 디바이스: {device.upper()}")
    
    # 2. TorchScript 모델 로드
    try:
        model_script = torch.jit.load(script_path, map_location=device)
        model_script.eval() # 평가 모드로 설정
        print(f"✅ TorchScript 모델 로드 완료: {script_path}")
    except Exception as e:
        print(f"❌ TorchScript 모델 로드 실패. 파일 경로를 확인해주세요: {e}")
        return

    # 3. 이미지 전처리 파이프라인 (학습 시와 동일한 전처리 사용)
    transform_A = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Tanh 출력이 [-1, 1]이므로, 입력도 동일하게 [-1, 1] 범위로 정규화합니다.
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    # 4. 이미지 변환 및 저장
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')])
    
    if not image_files:
        print(f"경고: 입력 폴더 '{input_dir}'에 변환할 이미지가 없습니다.")
        return
        
    print(f"'{input_dir}'에서 {len(image_files)}개 이미지 변환 시작...")

    with torch.no_grad(): # 추론 시에는 기울기 계산을 끕니다.
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            
            try:
                # 4.1. 이미지 로드 및 전처리
                image = Image.open(input_path).convert('RGB')
                
                # ⭐ 원본 크기 저장: (width, height) ⭐
                original_size = image.size 
                
                tensor_image = transform_A(image).unsqueeze(0).to(device) # [1, C, H, W] 배치 차원 추가

                # 4.2. 모델 추론
                output_tensor = model_script(tensor_image)

                # 4.3. 결과 후처리 (텐서 -> 이미지)
                # Tanh 출력 [-1, 1]을 [0, 1] 범위로 복원
                output_tensor = output_tensor.squeeze(0).cpu() # 배치 차원 제거
                output_tensor = (output_tensor * 0.5) + 0.5
                
                # [C, H, W] -> [H, W] (NIR 1채널, C=1)
                # squeeze(0)는 1채널 텐서의 채널 차원을 제거
                image_numpy = output_tensor.squeeze(0).numpy() * 255 
                
                # 4.4. PIL Image로 변환, 원본 크기로 복원 및 저장
                # 'L' 모드로 변환 (1채널 그레이스케일)
                output_pil = Image.fromarray(image_numpy.astype('uint8'), 'L')
                
                # ⭐ 원본 크기로 리사이즈하여 복원 ⭐
                output_pil = output_pil.resize(original_size, Image.Resampling.LANCZOS) # 고품질 Resampling 사용
                
                output_pil.save(os.path.join(output_dir, filename))
                
            except Exception as e:
                print(f"⚠️ 파일 처리 오류 발생: {filename} - {e}. 건너뜀.")
                continue
                
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"변환 중: {i+1}/{len(image_files)} 파일 저장 완료.")
    
    print(f"🎉 모든 이미지가 '{output_dir}'에 성공적으로 저장되었습니다.")


# ----------------------------------------------------
# 3. 메인 실행 블록
# ----------------------------------------------------
if __name__ == '__main__':
    # --- 설정 ---
    TORCHSCRIPT_FILE = "./models/rgb_to_nir_cut/model.pt" 
    INPUT_FOLDER = "./dataset/nsfw/image dataset/train/normal" # 변환할 RGB 이미지가 있는 폴더
    OUTPUT_FOLDER = "./dataset/nsfw/nir img dataset/train/normal" # 변환 결과를 저장할 폴더 (원본과 구분)

    # --- 실행 ---
    infer_from_torchscript(
        script_path=TORCHSCRIPT_FILE, 
        input_dir=INPUT_FOLDER, 
        output_dir=OUTPUT_FOLDER
    )
    